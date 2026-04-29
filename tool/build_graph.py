import os
import pickle
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from transformers import BertModel, BertTokenizer, AutoModel, AutoTokenizer
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#同小类内连边，冷门专利不连
@torch.no_grad()
def encode_texts(text_list, tokenizer, model, device, max_length=128, batch_size=32):
    model.to(device)
    model.eval()

    out = []
    for i in range(0, len(text_list), batch_size):
        batch = text_list[i:i + batch_size]
        enc = tokenizer(
            batch,
            add_special_tokens=True,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length
        ).to(device)

        outputs = model(**enc)
        cls = outputs.last_hidden_state[:, 0, :]
        out.append(cls.detach().cpu())

    if len(out) == 0:
        return torch.zeros(0, model.config.hidden_size)
    return torch.cat(out, dim=0)


def parse_ipc_cell(cell: str, keep_len=4):
    cell = (cell or "").strip()
    if not cell:
        return []

    if ";" in cell:
        parts = [x.strip() for x in cell.split(";") if x.strip()]
    elif "；" in cell:
        parts = [x.strip() for x in cell.split("；") if x.strip()]
    else:
        parts = [x.strip() for x in cell.split() if x.strip()]

    out = []
    for x in parts:
        if len(x) >= keep_len:
            out.append(x[:keep_len])

    return list(dict.fromkeys(out))

def load_ipc_desc(ipc_desc_path: str):
    df = pd.read_csv(ipc_desc_path, sep="\t", header=None)
    code = df[0].astype(str).str.strip()
    desc = df[2].astype(str).fillna("")
    return dict(zip(code, desc))

def build_mappings_and_texts(
    patent_tsv_path,
    cache_path="data/mappings.pkl",
):
    if os.path.exists(cache_path):
        print(f"[INFO] Loading mappings from cache: {cache_path}")
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        return cache["mappings"], cache["patent_text_dict"]

    print("[INFO] Cache not found, scanning patent TSV...")

    patent2idx = {}
    ipc2idx = {}
    patent_text_dict = {}

    patent_cnt = 0
    ipc_cnt = 0

    patent_ipc_edges = []
    patent2codes = defaultdict(list)
    code2patents = defaultdict(list)

    def get_patent_idx(pid):
        nonlocal patent_cnt
        if pid not in patent2idx:
            patent2idx[pid] = patent_cnt
            patent_cnt += 1
        return patent2idx[pid]

    def get_ipc_idx(code4):
        nonlocal ipc_cnt
        if code4 not in ipc2idx:
            ipc2idx[code4] = ipc_cnt
            ipc_cnt += 1
        return ipc2idx[code4]

    with open(patent_tsv_path, "r", encoding="utf-8") as f:
        header = f.readline().rstrip("\n").split("\t")
        if header:
            header[0] = header[0].lstrip("\ufeff")

        idx_id = header.index("id")
        idx_title = header.index("title")
        idx_abs = header.index("abstract")
        idx_ipc = header.index("xiaolei")

        for line in f:
            parts = line.rstrip("\n").split("\t")
            pid = str(parts[idx_id]).strip()

            p = get_patent_idx(pid)

            title = parts[idx_title].strip()
            abs_ = parts[idx_abs].strip()
            text = (title + " " + abs_).strip()
            if pid not in patent_text_dict:
                patent_text_dict[pid] = text

            codes4 = parse_ipc_cell(parts[idx_ipc], keep_len=4)
            for c4 in codes4:
                c = get_ipc_idx(c4)
                patent_ipc_edges.append((p, c))
                patent2codes[p].append(c)
                code2patents[c].append(p)

    # df：每个小类连接的不同专利数
    code_df = {c: len(set(plist)) for c, plist in code2patents.items()}

    mappings = {
        "patent2idx": patent2idx,
        "ipc2idx": ipc2idx,
        "patent_ipc_edges": patent_ipc_edges,
        "patent2codes": dict(patent2codes),
        "code_df": code_df,
    }

    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump({"mappings": mappings, "patent_text_dict": patent_text_dict}, f)

    print(f"[INFO] Saved mappings to: {cache_path}")
    print(f"[INFO] patents={len(patent2idx)} ipc={len(ipc2idx)} p-ipc edges={len(patent_ipc_edges)}")
    return mappings, patent_text_dict

def build_node_features(
    bert_path,
    device,
    mappings,
    patent_text_dict,
    ipc_desc_path,
    cache_file="data/node_features.pt",
    bert_bs_patent=32,
    bert_bs_ipc=32,
    max_len_patent=128,
    max_len_ipc=128,
):
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    if os.path.exists(cache_file):
        print(f"[INFO] Loading node features cache: {cache_file}")
        obj = torch.load(cache_file, map_location="cpu")
        return obj["patent_text_x"].float(), obj["ipc_x"].float(), obj["patent_x"].float()

    tokenizer = AutoTokenizer.from_pretrained(bert_path)
    model = AutoModel.from_pretrained(bert_path)

    patent2idx = mappings["patent2idx"]
    ipc2idx = mappings["ipc2idx"]

    # ---- patent texts by idx ----
    num_patent = len(patent2idx)
    id2pat = [None] * num_patent
    for pid, idx in patent2idx.items():
        id2pat[idx] = pid
    patent_texts = [patent_text_dict.get(pid, "") for pid in id2pat]

    # ---- ipc desc by idx ----
    ipc_desc_dict = load_ipc_desc(ipc_desc_path)

    num_ipc = len(ipc2idx)
    id2ipc = [None] * num_ipc
    for code4, idx in ipc2idx.items():
        id2ipc[idx] = code4
    ipc_texts = [ipc_desc_dict.get(code4, code4) for code4 in id2ipc]

    print("[INFO] Encoding patent_text_x ...")
    patent_text_x = encode_texts(
        patent_texts, tokenizer, model, device,
        max_length=max_len_patent, batch_size=bert_bs_patent
    )

    print("[INFO] Encoding ipc_x (subclass descriptions) ...")
    ipc_x = encode_texts(
        ipc_texts, tokenizer, model, device,
        max_length=max_len_ipc, batch_size=bert_bs_ipc
    )

    # ---- 聚合 mean(ipc_x) 到专利 ----
    px_edges = torch.tensor(mappings["patent_ipc_edges"], dtype=torch.long)
    p_idx = px_edges[:, 0]
    c_idx = px_edges[:, 1]

    # 用 CPU 聚合（省显存），也可放 GPU
    agg = torch.zeros(num_patent, ipc_x.size(-1), dtype=torch.float32)
    cnt = torch.zeros(num_patent, dtype=torch.float32)

    agg.index_add_(0, p_idx, ipc_x[c_idx].float())
    cnt.index_add_(0, p_idx, torch.ones_like(p_idx, dtype=torch.float32))
    mean_ipc = agg / cnt.clamp(min=1.0).unsqueeze(-1)

    patent_x = torch.cat([patent_text_x.float(), mean_ipc.float()], dim=-1)  # [Np,768]

    # 保存：float16 省空间
    torch.save(
        {
            "patent_text_x": patent_text_x,
            "ipc_x": ipc_x,
            "patent_x": patent_x,
            "meta": {
                "bert_path": bert_path,
                "max_len_patent": max_len_patent,
                "max_len_ipc": max_len_ipc,
                "patent_x_rule": "patent_text_x + mean(ipc_x)",
            }
        },
        cache_file
    )
    print(f"[INFO] Saved node features to: {cache_file}")
    return patent_text_x.float(), ipc_x.float(), patent_x.float()


def build_pp_edges_group_topk_union(
    vec_for_sim_cpu_f32: torch.Tensor,
    mappings,
    out_path="data/pp_edges.pt",
    topk_per_patent=20,          # 最终每篇专利最多保留多少条 p->q
    min_df=5,                    # 冷门小类阈值：df < min_df 不建边
    hot_df=20000,                # 热门小类阈值：df >= hot_df 视为热门
    k_per_class=10,              # 普通小类：每个专利在该小类内取 top k_per_class
    k_hot=3,                     # 热门小类：每个专利在该小类内只取 top k_hot（更严格/更少）
    min_cos=0.0,                 # 可选：相似度阈值（建议 0.4~0.7 之间试）
    exact_threshold=2000,        # 小类规模 <= 该值：用精确 IndexFlatIP；否则用 HNSW
    hnsw_m=32,
    ef_search=128,
    batch=2048,                  # 小类内检索时的query batch
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        print(f"[INFO] Loading cached p-p edges: {out_path}")
        obj = torch.load(out_path, map_location="cpu")
        return obj["pp_edge_index"]

    import faiss

    patent2codes = mappings["patent2codes"]   # p -> list[code_idx]
    code_df = mappings["code_df"]             # code_idx -> df

    # 1) 归一化向量（余弦=内积）
    X = vec_for_sim_cpu_f32.detach().cpu().numpy().astype("float32")
    faiss.normalize_L2(X)
    N, D = X.shape

    # 2) 建 code -> patents（从 patent2codes 反推，避免你缓存里没有 code2patents）
    code2patents = defaultdict(list)
    for p, codes in patent2codes.items():
        for c in codes:
            code2patents[c].append(p)

    # 3) 收集所有候选边（可能重复，后面 union+截断）
    src_all = []
    dst_all = []
    sim_all = []

    # 4) 遍历每个小类，做组内 topK 检索
    codes_sorted = sorted(code2patents.keys())
    print(f"[INFO] Build p-p edges by subclass topK. total subclasses={len(codes_sorted)}")

    for c in codes_sorted:
        plist = code2patents[c]
        if not plist:
            continue

        # 去重
        plist = list(dict.fromkeys(plist))
        df = len(plist)

        # 冷门：跳过
        if df < min_df:
            continue

        # 热门：只取更少的 topK
        k = k_hot if df >= hot_df else k_per_class
        if k <= 0:
            continue

        # 组内向量
        V = X[plist]  # [df, D]

        # 选择索引类型
        if df <= exact_threshold:
            index = faiss.IndexFlatIP(D)  # 精确内积（余弦）
        else:
            index = faiss.IndexHNSWFlat(D, hnsw_m)
            index.hnsw.efSearch = ef_search

        index.add(V)

        # 分批检索（避免一次性占太多内存）
        # 搜索 k+1 是为了去掉自己
        for st in range(0, df, batch):
            ed = min(df, st + batch)
            q = V[st:ed]
            Dv, Iv = index.search(q, k + 1)

            # 逐条写边
            for i in range(ed - st):
                p_global = plist[st + i]
                # Dv[i] 是相似度（内积=cos），Iv[i] 是在 plist 内的局部下标
                kept = 0
                for score, j_local in zip(Dv[i], Iv[i]):
                    if j_local < 0:
                        continue
                    q_global = plist[int(j_local)]
                    if q_global == p_global:
                        continue
                    if score < min_cos:
                        continue

                    src_all.append(p_global)
                    dst_all.append(q_global)
                    sim_all.append(float(score))
                    kept += 1
                    if kept >= k:
                        break

    if len(src_all) == 0:
        pp_edge_index = torch.zeros((2, 0), dtype=torch.long)
        torch.save({"pp_edge_index": pp_edge_index, "meta": {"empty": True}}, out_path)
        print(f"[WARN] No p-p edges generated. saved empty to {out_path}")
        return pp_edge_index

    # 5) union + 截断：按 (src, -sim) 排序，然后每个 src 去重 dst，保留 topk_per_patent
    src_np = np.asarray(src_all, dtype=np.int64)
    dst_np = np.asarray(dst_all, dtype=np.int64)
    sim_np = np.asarray(sim_all, dtype=np.float32)

    # sort by src asc, sim desc
    order = np.lexsort((-sim_np, src_np))
    src_np = src_np[order]
    dst_np = dst_np[order]
    sim_np = sim_np[order]

    final_src = []
    final_dst = []

    cur_src = -1
    used = set()
    kept = 0

    for s, d in zip(src_np, dst_np):
        if s != cur_src:
            cur_src = s
            used.clear()
            kept = 0
        if kept >= topk_per_patent:
            continue
        if d in used:
            continue
        used.add(int(d))
        final_src.append(int(s))
        final_dst.append(int(d))
        kept += 1

    pp_edge_index = torch.tensor([final_src, final_dst], dtype=torch.long)  # [2, E] 有向 p->q

    torch.save(
        {
            "pp_edge_index": pp_edge_index,
            "meta": {
                "strategy": "subclass_intra_topk_union_truncate",
                "topk_per_patent": topk_per_patent,
                "min_df": min_df,
                "hot_df": hot_df,
                "k_per_class": k_per_class,
                "k_hot": k_hot,
                "min_cos": min_cos,
                "exact_threshold": exact_threshold,
                "hnsw_m": hnsw_m,
                "ef_search": ef_search,
                "directed": True,
                "candidates_before_union": int(len(src_all)),
                "edges_after_union": int(pp_edge_index.size(1)),
            }
        },
        out_path
    )
    print(f"[INFO] Saved p-p edges to: {out_path}, E={pp_edge_index.size(1)} "
          f"(before_union={len(src_all)})")
    return pp_edge_index



# -----------------------------
# 7) 总入口：从零构图 + 特征 + 边
# -----------------------------
def build_all(
    patent_tsv_path,
    ipc_desc_path,
    out_dir="data",
    bert_path="bert-base-uncased",
    device="cuda",
):
    os.makedirs(out_dir, exist_ok=True)

    mappings_path = os.path.join(out_dir, "mappings.pkl")
    feats_path = os.path.join(out_dir, "node_features.pt")
    pp_path = os.path.join(out_dir, "pp_edges.pt")
    struct_path = os.path.join(out_dir, "graph_structure.pkl")

    # 1) mappings + 文本 + 专利-ipc 边
    mappings, patent_text_dict = build_mappings_and_texts(
        patent_tsv_path,
        cache_path=mappings_path
    )

    # 2) 节点特征
    patent_text_x, ipc_x, patent_x = build_node_features(
        bert_path=bert_path,
        device=device,
        mappings=mappings,
        patent_text_dict=patent_text_dict,
        ipc_desc_path=ipc_desc_path,
        cache_file=feats_path
    )

    # 3) p-ipc edge_index
    px = torch.tensor(mappings["patent_ipc_edges"], dtype=torch.long).t().contiguous()

    # 4) p-p edges: 全局ANN + 共享小类过滤
    # 默认用纯专利文本向量做 ANN（更符合“相似度只用专利文本”）
    vec_for_sim = patent_text_x.detach().cpu().float()

    pp = build_pp_edges_group_topk_union(
        vec_for_sim_cpu_f32=vec_for_sim,
        mappings=mappings,
        out_path=pp_path,
        topk_per_patent=20,  # 最终每篇最多 20 个邻居
        min_df=5,  # 冷门小类跳过
        hot_df=20000,  # 热门小类阈值（你可按统计调）
        k_per_class=20,  # 普通小类组内 top10
        k_hot=20,  # 热门小类只取 top3
        min_cos=0.6,  # 可调：比如 0.5/0.6 能更干净
        exact_threshold=2000,  # 小类<=2000用精确；更大用HNSW
        hnsw_m=32,
        ef_search=128,
        batch=2048
    )

    edge_index_dict = {
        ("patent", "has_ipc", "ipc"): px,
        ("ipc", "rev_has_ipc", "patent"): px.flip(0),
        ("patent", "sim", "patent"): pp,
        ("patent", "rev_sim", "patent"): pp.flip(0),
    }

    # 5) 保存结构（方便你直接喂给 PyG）
    with open(struct_path, "wb") as f:
        pickle.dump(
            {
                "edge_index_dict": edge_index_dict,
                "num_patent": patent_x.size(0),
                "num_ipc": ipc_x.size(0),
            },
            f
        )

    print("[INFO] DONE.")
    print(f"  mappings : {mappings_path}")
    print(f"  features : {feats_path}  (contains patent_text_x, ipc_x, patent_x)")
    print(f"  pp edges : {pp_path}")
    print(f"  structure: {struct_path}")


if __name__ == "__main__":
    PATENT_TSV = "/home/pc3/zy/data/src/TOPK_NOIPC/patent.tsv"
    IPC_DESC_TSV = "/home/pc3/zy/data/src/TOPK_NOIPC/ipc.tsv"

    build_all(
        patent_tsv_path=PATENT_TSV,
        ipc_desc_path=IPC_DESC_TSV,
        out_dir="/home/pc3/zy/second/data",
        bert_path="/home/pc3/zy/premodel/bge-base-en-v1.5",
        device="cuda" if torch.cuda.is_available() else "cpu"
    )
