import torch
import pickle
import numpy as np
from torch import nn
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer
from torch_geometric.data import HeteroData
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
from torch_geometric.loader import NeighborLoader


class HeteroGraphSAGE(nn.Module):
    def __init__(
        self,
        metadata,
        in_dims: dict,
        hidden_dim=256,
        out_dim=256,
        num_layers=2,
        dropout=0.2,
        rel_aggr="sum",
        sage_aggr="mean",
        use_out_ln=True,
    ):
        super().__init__()
        self.metadata = metadata
        self.dropout = dropout
        self.num_layers = num_layers
        self.use_out_ln = use_out_ln

        node_types, edge_types = metadata

        self.lin_dict = nn.ModuleDict({
            ntype: Linear(in_dims[ntype], hidden_dim)
            for ntype in node_types
        })

        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            conv_dict = {}
            for etype in edge_types:
                conv_dict[etype] = SAGEConv(
                    (hidden_dim, hidden_dim),
                    hidden_dim,
                    aggr=sage_aggr,
                    normalize=False
                )
            self.convs.append(HeteroConv(conv_dict, aggr=rel_aggr))

        self.out_lin = nn.ModuleDict({
            ntype: Linear(hidden_dim, out_dim)
            for ntype in node_types
        })

        if self.use_out_ln:
            self.out_ln = nn.ModuleDict({
                ntype: nn.LayerNorm(out_dim)
                for ntype in node_types
            })
        else:
            self.out_ln = None

    def forward_backbone(self, x_dict, edge_index_dict):
        h = {}
        for k, x in x_dict.items():
            h[k] = F.gelu(self.lin_dict[k](x))
            h[k] = F.dropout(h[k], p=self.dropout, training=self.training)

        for conv in self.convs:
            h_new = conv(h, edge_index_dict)
            for k in h.keys():
                if k not in h_new or h_new[k] is None:
                    h_new[k] = h[k]

            h = {k: F.gelu(v) for k, v in h_new.items()}
            h = {k: F.dropout(v, p=self.dropout, training=self.training) for k, v in h.items()}

        return h

    def forward_head(self, h_dict):
        out = {k: self.out_lin[k](v) for k, v in h_dict.items()}
        if self.out_ln is not None:
            out = {k: self.out_ln[k](v) for k, v in out.items()}
        return out

    def forward(self, x_dict, edge_index_dict):
        h = self.forward_backbone(x_dict, edge_index_dict)
        out = self.forward_head(h)
        return out


class net(nn.Module):
    def __init__(self, args):
        super(net, self).__init__()
        self.args = args
        self.device = args.device

        # ---------------- BERT ----------------
        self.tokenizer = BertTokenizer.from_pretrained(args.bert_path)
        self.bert_shared = BertModel.from_pretrained(args.bert_path)

        hidden_size = 768

        # ---------------- 图数据路径 ----------------
        self.graph_feat_path = getattr(args, "graph_feat_path", "data/node_features.pt")
        self.graph_struct_path = getattr(args, "graph_struct_path", "data/graph_structure.pkl")
        self.mappings_path = getattr(args, "mappings_path", "data/mappings.pkl")

        with open(self.mappings_path, "rb") as f:
            cache = pickle.load(f)
        mappings = cache["mappings"]
        self.patent2idx = mappings["patent2idx"]

        # ---------------- 节点初始特征 ----------------
        feat = torch.load(self.graph_feat_path, map_location="cpu")
        patent_x0 = feat["patent_text_x"].float()
        ipc_x0 = feat["ipc_x"].float()

        self.num_patent = patent_x0.size(0)
        self.num_ipc = ipc_x0.size(0)

        self.register_buffer("patent_x0", patent_x0)
        self.register_buffer("ipc_x0", ipc_x0)

        # ---------------- 边结构 ----------------
        with open(self.graph_struct_path, "rb") as f:
            struct = pickle.load(f)
        edge_index_dict = struct["edge_index_dict"]
        self.edge_index_dict = {k: v.cpu() for k, v in edge_index_dict.items()}

        # ---------------- GraphSAGE 超参数 ----------------
        self.gnn_hidden = getattr(args, "gnn_hidden", 768)
        self.gnn_out = getattr(args, "gnn_out", 768)
        self.gnn_layers = getattr(args, "gnn_layers", 2)
        self.gnn_dropout = getattr(args, "gnn_dropout", 0.1)
        self.rel_aggr = getattr(args, "rel_aggr", "sum")
        self.sage_aggr = getattr(args, "sage_aggr", "mean")

        node_types = ["patent", "ipc"]
        edge_types = list(self.edge_index_dict.keys())
        metadata = (node_types, edge_types)

        in_dims = {
            "patent": self.patent_x0.size(-1),
            "ipc": self.ipc_x0.size(-1),
        }

        self.gnn = HeteroGraphSAGE(
            metadata=metadata,
            in_dims=in_dims,
            hidden_dim=self.gnn_hidden,
            out_dim=self.gnn_out,
            num_layers=self.gnn_layers,
            dropout=self.gnn_dropout,
            rel_aggr=self.rel_aggr,
            sage_aggr=self.sage_aggr
        )
        for p in self.gnn.parameters():
            p.requires_grad = True

        self.use_neighbor_sampling = getattr(args, "use_neighbor_sampling", True)
        default_neighbors = [15] * self.gnn_layers
        self.num_neighbors = getattr(args, "num_neighbors", default_neighbors)

        # 预构建 HeteroData
        self.data = self._build_heterodata_cpu()

        # 固定采样 generator
        self._sample_gen = torch.Generator(device="cpu")
        self._sample_gen.manual_seed(getattr(args, "seed", 42))

        # ---------------- 分解 ----------------
        self.shared_proj = nn.Linear(hidden_size, hidden_size)
        self.gate_layer = nn.Linear(hidden_size, hidden_size * 3)
        self.REP_I = nn.Sequential(*self.rep_layer(input_dims=hidden_size, out_dims=hidden_size, layer=2))
        self.REP_C = nn.Sequential(*self.rep_layer(input_dims=hidden_size, out_dims=hidden_size, layer=2))
        self.REP_A = nn.Sequential(*self.rep_layer(input_dims=hidden_size, out_dims=hidden_size, layer=2))
        self.map_t = nn.Sequential(*self.rep_layer(input_dims=1, out_dims=64, layer=2))

        # ---------------- 输出头 ----------------
        self.t_regress_c = nn.Sequential(*self.output_layer(input_dims=hidden_size, out_dims=2, layer=4))
        self.t_regress_i = nn.Sequential(*self.output_layer(input_dims=hidden_size, out_dims=2, layer=4))

        self.y_regress = nn.Sequential(*self.output_layer(
            input_dims=hidden_size * 2 + 64 + 768 + 768,
            out_dims=2,
            layer=5
        ))

        self.y_regress_a = nn.Sequential(*self.output_layer(input_dims=hidden_size, out_dims=2, layer=4))
        self.y_regress_c = nn.Sequential(*self.output_layer(input_dims=hidden_size, out_dims=2, layer=4))

        #---------------- 图变换 -------------------
        self.gnn_pair_proj = nn.Sequential(*self.rep_layer(4 * self.gnn_out, 768, layer=4))

        #参考系
        self.ref_payload_path = getattr(args, "ref_payload_path", "ref_emb_top10.pt")
        ref_payload = torch.load(self.ref_payload_path, map_location="cpu")

        self.ref_topk_map = ref_payload["topk_map"]
        ref_emb_dict = ref_payload["ref_emb"]
        self.ref_k = int(ref_payload.get("k", 10))

        # table
        ref_ids = list(ref_emb_dict.keys())
        ref_mat = torch.stack([ref_emb_dict[rid].float() for rid in ref_ids], dim=0)
        self.register_buffer("ref_emb_table", ref_mat)
        self.refid2row = {str(rid).strip(): i for i, rid in enumerate(ref_ids)}

        K = self.ref_k
        w = torch.arange(K, dtype=torch.float32)
        w = torch.exp(-w)
        w = w / (w.sum() + 1e-9)
        self.register_buffer("ref_rank_w", w)

        # evidence温度
        self.ref_tau = float(getattr(args, "ref_tau", 0.3))
        self.ref_aug_proj = nn.Sequential(*self.rep_layer(self.ref_emb_table.size(1) + 3, 768, layer=3))
        self.ref_gate = nn.Sequential(
            nn.Linear(3, 64),
            nn.Tanh(),
            nn.Dropout(p=getattr(args, "dropout", 0.1)),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def _build_heterodata_cpu(self):
        data = HeteroData()
        data["patent"].x = self.patent_x0.cpu()
        data["ipc"].x = self.ipc_x0.cpu()
        for etype, eidx in self.edge_index_dict.items():
            data[etype].edge_index = eidx
        return data

    # -------------------- BERT 编码 --------------------
    def bert_pair(self, text_a, text_b):
        enc = self.tokenizer(
            text_a, text_b,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        ).to(self.args.device)
        outputs = self.bert_shared(**enc)
        return outputs.last_hidden_state[:, 0, :]

    def bert_single(self, text):
        enc = self.tokenizer(
            text,
            add_special_tokens=True,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=128
        ).to(self.args.device)
        outputs = self.bert_shared(**enc)
        return outputs.last_hidden_state[:, 0, :]

    def _get_ref_rows_for_A(self, patentA_ids):
        B = len(patentA_ids)
        K = self.ref_k
        ref_rows = torch.full((B, K), -1, dtype=torch.long)

        for i, pid in enumerate(patentA_ids):
            aid = str(pid).strip()
            refs = self.ref_topk_map.get(aid, [])[:K]
            for j, rid in enumerate(refs):
                ref_rows[i, j] = self.refid2row.get(str(rid).strip(), -1)

        return ref_rows.to(self.device)

    def _ref_aug_from_A_to_B(self, patentA_ids, b_emb):
        B = len(patentA_ids)
        H = self.ref_emb_table.size(1)
        K = self.ref_k

        ref_rows = self._get_ref_rows_for_A(patentA_ids)
        valid = ref_rows >= 0

        emb = torch.zeros(B, K, H, device=self.device)
        if valid.any():
            emb[valid] = self.ref_emb_table[ref_rows[valid]]

        w = self.ref_rank_w[:K].to(self.device).view(1, K, 1)
        ref_pool = (emb * w).sum(dim=1)
        ref_pool = F.normalize(ref_pool, p=2, dim=-1)

        b = F.normalize(b_emb, p=2, dim=-1).unsqueeze(1)
        sims = (emb * b).sum(dim=-1)

        sims_masked = sims.masked_fill(~valid, float("-inf"))

        tau = max(self.ref_tau, 1e-6)
        att = torch.softmax(sims_masked / tau, dim=1)
        softmean = (att * sims.masked_fill(~valid, 0.0)).sum(dim=1)

        sims0 = sims.masked_fill(~valid, 0.0)
        denom = valid.sum(dim=1).clamp_min(1).float()
        mean = sims0.sum(dim=1) / denom

        var = ((sims0 - mean.unsqueeze(1)) ** 2).masked_fill(~valid, 0.0).sum(dim=1) / denom
        std = torch.sqrt(var + 1e-8)

        evidence = torch.stack([softmean, mean, std], dim=1)
        evidence = torch.nan_to_num(evidence, neginf=0.0, posinf=0.0)

        ref_aug = torch.cat([ref_pool, evidence], dim=1)
        ref_aug = self.ref_aug_proj(ref_aug)

        g = self.ref_gate(evidence)
        ref_aug = ref_aug * g

        return ref_aug, evidence, g

    # -------------------- 分解 --------------------
    def decompose_with_gate(self, text_emb):
        shared = self.shared_proj(text_emb)

        B, H = shared.shape
        gates_logits = self.gate_layer(shared).view(B, H, 3)
        gates = torch.softmax(gates_logits, dim=-1)

        w_c = gates[:, :, 0]
        w_a = gates[:, :, 1]
        w_i = gates[:, :, 2]

        rep_c_base = w_c * shared
        rep_a_base = w_a * shared
        rep_i_base = w_i * shared

        rep_c = self.REP_C(rep_c_base)
        rep_a = self.REP_A(rep_a_base)
        rep_i = self.REP_I(rep_i_base)

        recon = rep_c + rep_a + rep_i
        return shared, gates, rep_c, rep_a, rep_i, recon

    # -------------------- 图路：pair --------------------
    def forward_graph_pair(self, p1_idx: torch.Tensor, p2_idx: torch.Tensor):
        valid1 = (p1_idx >= 0)
        valid2 = (p2_idx >= 0)
        seeds = torch.unique(torch.cat([p1_idx[valid1], p2_idx[valid2]], dim=0)).to("cpu")

        if not self.use_neighbor_sampling:
            x_dict = {
                "patent": self.patent_x0.to(self.device),
                "ipc": self.ipc_x0.to(self.device),
            }
            edge_index_dict = {k: v.to(self.device) for k, v in self.edge_index_dict.items()}
            h_hidden = self.gnn.forward_backbone(x_dict, edge_index_dict)
            out = self.gnn.forward_head(h_hidden)

            h_patent = out["patent"]
            h1 = torch.where(
                valid1.unsqueeze(-1),
                h_patent[p1_idx.clamp_min(0)],
                torch.zeros_like(h_patent[p1_idx.clamp_min(0)])
            )
            h2 = torch.where(
                valid2.unsqueeze(-1),
                h_patent[p2_idx.clamp_min(0)],
                torch.zeros_like(h_patent[p2_idx.clamp_min(0)])
            )
            return h1, h2

        loader = NeighborLoader(
            self.data,
            input_nodes=("patent", seeds),
            num_neighbors=self.num_neighbors,
            batch_size=seeds.numel(),
            shuffle=False,
            num_workers=0,
            persistent_workers=False,
            generator=self._sample_gen,
        )
        sampled = next(iter(loader)).to(self.device)

        n_id = sampled["patent"].n_id.cpu().tolist()
        id2local = {gid: i for i, gid in enumerate(n_id)}

        def build_local(p_idx: torch.Tensor):
            B = p_idx.size(0)
            local = torch.full((B,), -1, dtype=torch.long, device=self.device)
            p_cpu = p_idx.cpu().tolist()
            for i, pid in enumerate(p_cpu):
                if pid >= 0 and pid in id2local:
                    local[i] = id2local[pid]
            return local

        local1 = build_local(p1_idx)
        local2 = build_local(p2_idx)

        x_dict = {
            "patent": sampled["patent"].x,
            "ipc": sampled["ipc"].x,
        }
        edge_index_dict = {k: sampled[k].edge_index for k in self.edge_index_dict.keys()}

        h_hidden = self.gnn.forward_backbone(x_dict, edge_index_dict)
        out = self.gnn.forward_head(h_hidden)

        def gather_from_local(local: torch.Tensor, valid_mask: torch.Tensor):
            B = local.size(0)
            h = torch.zeros(B, out["patent"].size(-1), device=self.device)
            ok = (local >= 0) & valid_mask
            if ok.any():
                h[ok] = out["patent"][local[ok]]
            return h

        h1 = gather_from_local(local1, valid1)
        h2 = gather_from_local(local2, valid2)
        return h1, h2

    def _patent_id_to_idx(self, patent_ids):
        out = []
        for pid in patent_ids:
            pid = str(pid).strip()
            out.append(self.patent2idx.get(pid, -1))
        return torch.tensor(out, dtype=torch.long)

    def forward(self, input_data):
        # 1) pair文本
        text_emb = self.bert_pair(input_data['text_a'], input_data['text_b'])  # [B,768]
        rep_t = self.map_t(input_data['xiaolei'].unsqueeze(1))
        shared, gate, rep_c, rep_a, rep_i, recon = self.decompose_with_gate(text_emb)
        h_bert_pair = torch.cat((rep_c, rep_a, rep_t), dim=1)  # [B,1600]

        # 2) 图
        p1 = self._patent_id_to_idx(input_data["patentA"]).to(self.device)
        p2 = self._patent_id_to_idx(input_data["patentB"]).to(self.device)
        input_data["patentA_idx"] = p1
        input_data["patentB_idx"] = p2

        h1, h2 = self.forward_graph_pair(p1, p2)
        h_graph_pair = torch.cat([h1, h2, torch.abs(h1 - h2), h1 * h2], dim=-1)
        h_graph_pair = self.gnn_pair_proj(h_graph_pair)

        # 3) 参考系
        b_emb = self.bert_single(input_data["text_b"])
        ref_aug, ref_evidence, ref_g = self._ref_aug_from_A_to_B(input_data["patentA"], b_emb)

        # 4) 融合输出
        y_input = torch.cat((h_bert_pair, h_graph_pair, ref_aug), dim=1)
        pred_y = self.y_regress(y_input)[:, 1]

        a_pred_y = self.y_regress_a(rep_a)[:, 1]
        c_pred_y = self.y_regress_c(rep_c)[:, 1]
        c_pred_t = self.t_regress_c(rep_c)[:, 1]
        i_pred_t = self.t_regress_i(rep_i)[:, 1]

        output_data = {
            'text_emb': text_emb,
            'shared': shared,
            'gate': gate,
            'recon': recon,
            'rep_i': rep_i,
            'rep_c': rep_c,
            'rep_a': rep_a,
            'pred_y': pred_y,
            'a_pred_y': a_pred_y,
            'c_pred_y': c_pred_y,
            'c_pred_t': c_pred_t,
            'i_pred_t': i_pred_t,
            'ref_aug': ref_aug,
            'ref_evidence': ref_evidence,
            'ref_gate_scalar': ref_g,
        }

        self.output = output_data
        self.input = input_data
        return output_data

    # loss
    def loss_func(self):
        out = self.output
        inp = self.input

        a_loss_y = F.binary_cross_entropy(input=out['a_pred_y'], target=inp['label'], reduction='mean')
        c_loss_y = F.binary_cross_entropy(input=out['c_pred_y'], target=inp['label'], reduction='mean')
        loss_y = F.binary_cross_entropy(input=out['pred_y'], target=inp['label'], reduction='mean')
        c_loss_t = F.binary_cross_entropy(input=out['c_pred_t'], target=inp['xiaolei'], reduction='mean')
        i_loss_t = F.binary_cross_entropy(input=out['i_pred_t'], target=inp['xiaolei'], reduction='mean')

        loss_recon = F.mse_loss(out['recon'], out['shared'])
        loss_orth = (
            self.orthogonal_feature_loss(out['rep_c'], out['rep_a']) +
            self.orthogonal_feature_loss(out['rep_c'], out['rep_i']) +
            self.orthogonal_feature_loss(out['rep_a'], out['rep_i'])
        ) / 3.0

        if self.input['tr']:
            loss = c_loss_t + i_loss_t + loss_y + a_loss_y + c_loss_y + loss_recon + loss_orth
        else:
            loss = loss_y

        logs = {
            'loss_y': loss_y.detach(),
            'a_loss_y': a_loss_y.detach(),
            'c_loss_y': c_loss_y.detach(),
            'loss_recon': loss_recon.detach(),
            'loss_orth': loss_orth.detach(),
            'loss_total': loss.detach(),
            'rank_loss': torch.tensor(0.0, device=loss.device),
        }
        return loss, logs

    def rep_layer(self, input_dims, out_dims, layer):
        dim = np.around(np.linspace(input_dims, out_dims, layer + 1)).astype(int)
        layers = []
        for i in range(layer):
            layers.append(nn.Linear(dim[i], dim[i + 1]))
            layers.append(nn.Tanh())
            layers.append(nn.Dropout(p=self.args.dropout))
        return layers

    def output_layer(self, input_dims, out_dims, layer):
        dim = np.around(np.linspace(input_dims, out_dims, layer + 1)).astype(int)
        layers = []
        for i in range(layer):
            layers.append(nn.Linear(dim[i], dim[i + 1]))
            if i < layer - 1:
                layers.append(nn.Tanh())
                layers.append(nn.Dropout(p=self.args.dropout))
        layers.append(nn.Softmax(dim=1))
        return layers

    def orthogonal_feature_loss(self, rep_c, rep_a):
        rc = rep_c - rep_c.mean(dim=0, keepdim=True)
        ra = rep_a - rep_a.mean(dim=0, keepdim=True)

        B = rc.size(0)
        cov = torch.matmul(rc.t(), ra) / (B + 1e-8)
        return (cov ** 2).mean()
