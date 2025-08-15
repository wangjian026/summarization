"""

"""
import ast
import json
from typing import List, Tuple, Dict, Any
import numpy as np
import itertools
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from EviSum.upload.argosum_EO import preprocess
from largemodel import llm

# NLI
tokenizer = AutoTokenizer.from_pretrained("../pretrainmodels/roberta-large-mnli")
model = AutoModelForSequenceClassification.from_pretrained("../pretrainmodels/roberta-large-mnli")
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


def try_parse_json(response_str):
    try:
        data=json.loads(response_str)
        for key, value in data.items():
            if "aspect" in str(key).lower():
                return None, False
            else:
                return json.loads(response_str), True
    except json.JSONDecodeError:
        return None, False



def call_llm_extract_aspects(client_socket_sum,Dm: List[str], Dv: List[str],get_aspect_prompt) -> Tuple[List[str], Dict[str, str]]:
    """

    return:
      - aspects: List of aspect ids / names (e.g. ["price","service",...])
      - I_texts: dict mapping aspect -> statement text
    """
    # aspects = ["aspect_1", "aspect_2", "aspect_3"]
    # I_texts = {a: f"This is a synthesized aspect statement for {a}." for a in aspects}

    max_retries = 10
    for attempt in range(max_retries):
        response = llm.getAnswer(client_socket_sum,query=get_aspect_prompt)
        # print('response',response)
        data, success = try_parse_json(response)
        if success:

            break
        else:

            print(f"⚠️ failed {attempt + 1} …")
            attempt=attempt+1

    Aspects = []
    Aspect_explantion_list = []
    I_texts={}
    for key, value in data.items():
        Aspects.append(key)
        Aspect_explantion_list.append(value)
        I_texts[key] = value

    assert len(Aspects) == len(Aspect_explantion_list)

    return Aspects, I_texts

def nli_entailment_score(premise: str, hypothesis: str) -> float:
    """

    """
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze()
    entail_score = probs[2].item()  # entailment
    return entail_score

def nli_contradiction_score(premise: str, hypothesis: str) -> float:
    """

    """
    inputs = tokenizer(premise, hypothesis, return_tensors="pt", truncation=True, padding=True).to(device)
    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1).squeeze()
    contradict_score = probs[0].item()  # contradiction
    return contradict_score  #

# ---------------------------
# ---------------------------

def compute_embeddings(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    """
    try:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name)
        emb = model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
        return emb
    except Exception as e:
        rng = np.random.default_rng(0)
        dim = 384
        emb = rng.normal(size=(len(texts), dim)).astype(float)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(min=1e-12)
        return emb

# ---------------------------
# ----  -----
# ---------------------------

class EvidencePairSelector:
    def __init__(
        self,
        Dm: List[str],
        Dv: List[str],
        o_m: str,
        o_v: str,
        client_socket_sum,
        get_aspect_prompt: str,
        lambda_: float = 2.0,
        alpha: float = 0.5,
        m: int = 10,
        drop_ratio: float =0.3,
        embedding_model: str = "all-MiniLM-L6-v2",
    ):
        """
        初始化选择器
        """
        self.Dm = Dm
        self.Dv = Dv
        self.o_m = o_m
        self.o_v = o_v
        self.lambda_ = lambda_
        self.alpha = alpha
        self.m = m
        self.drop_ratio = drop_ratio
        self.embedding_model = embedding_model
        self.get_aspect_prompt = get_aspect_prompt
        # 1) extract aspects
        self.aspects, self.I_texts = call_llm_extract_aspects(client_socket_sum,Dm, Dv,get_aspect_prompt)


        # 2)  embeddings
        self.texts_for_embed = Dm + Dv + [self.I_texts[f] for f in self.aspects]
        self.embeddings = compute_embeddings(self.texts_for_embed, model_name=self.embedding_model)
        self.dim = self.embeddings.shape[1]

        self.emb_dm = self.embeddings[:len(Dm)]
        self.emb_dv = self.embeddings[len(Dm):len(Dm)+len(Dv)]
        self.emb_I = self.embeddings[len(Dm)+len(Dv):]  # per aspect order
        o_m_vec = compute_embeddings([self.o_m], model_name=self.embedding_model)[0]
        o_v_vec = compute_embeddings([self.o_v], model_name=self.embedding_model)[0]


        # 3)  Q(pair) 和 rel(pair, f)
        self.pairs = list(itertools.product(range(len(Dm)), range(len(Dv))))  # pairs as index tuples
        self.num_pairs = len(self.pairs)
        self.num_aspects = len(self.aspects)

        # Q values: length num_pairs
        self.Q_vals = np.zeros(self.num_pairs, dtype=float)
        # rels: shape (num_pairs, num_aspects)
        self.rels = np.zeros((self.num_pairs, self.num_aspects), dtype=float)

        cos_I_f_o_m = np.dot(self.emb_I, o_m_vec) / (
                np.linalg.norm(self.emb_I, axis=1) * np.linalg.norm(o_m_vec)
        )

        # sim between aspect and opinion
        cos_I_f_o_v = np.dot(self.emb_I, o_v_vec) / (
                np.linalg.norm(self.emb_I, axis=1) * np.linalg.norm(o_v_vec)
        )
        self.aspect_weights = np.maximum(0, (cos_I_f_o_m + cos_I_f_o_v) / 2) # shape: (num_aspects,)


        self._precompute_pair_scores()

    def _precompute_pair_scores(self):
        """
        """
        # Precompute sup scores and contra scores for speed ()
        # In practice: replace by batched calls to a NLI model
        sup_a_vs_om = [nli_entailment_score(a, self.o_m) for a in self.Dm]
        sup_b_vs_ov = [nli_entailment_score(b, self.o_v) for b in self.Dv]

        # Optionally precompute contradiction for each (a,b)
        contra_pairs = [nli_contradiction_score(self.Dm[i], self.Dv[j]) for (i, j) in self.pairs]

        # Compute rel((a,b), f) = cosine(a, I(f)) * cosine(b, I(f))
        # cosines: reuse normalized embeddings and dot product
        # emb_dm shape (Nd1, d), emb_dv shape (Nd2, d), emb_I shape (Na, d)
        # compute cosine(a, I) for all a,I -> (Nd1, Na)
        cos_a_I = self.emb_dm @ self.emb_I.T  # dot of normalized vectors
        cos_b_I = self.emb_dv @ self.emb_I.T  # (Nd2, Na)

        for idx, (i_a, j_b) in enumerate(self.pairs):
            sup_a = sup_a_vs_om[i_a]
            sup_b = sup_b_vs_ov[j_b]
            contra = contra_pairs[idx]
            self.Q_vals[idx] = self.alpha * (sup_a * sup_b) + (1.0 - self.alpha) * contra

            # rel per aspect:
            # rel((a,b), f) = cos(a,I_f) * cos(b,I_f)
            # cos_a_I[i_a, :] -> (Na,)
            # cos_b_I[j_b, :] -> (Na,)
            # self.rels[idx, :] = cos_a_I[i_a, :] * cos_b_I[j_b, :]
            self.rels[idx, :] = np.maximum(0, cos_a_I[i_a, :] * cos_b_I[j_b, :])


    def greedy_select(self) -> Dict[str, Any]:
        """
        max F(E)=Q(E)+lambda*AspCover(E)
        return selected_pairs, selected_score, history,
        """
        selected = []
        selected_mask = np.zeros(self.num_pairs, dtype=bool)
        curr_sum_rel = np.zeros(self.num_aspects, dtype=float)
        curr_score = 0.0

        history = []
        prev_gain = None
        # for step in range(self.num_pairs):

        for step in range(min(self.m, self.num_pairs)):
            best_idx = None
            best_gain = -np.inf

            #
            remain_idxs = np.where(~selected_mask)[0]
            # AspCover gains for all remaining:
            # delta_f = sqrt(curr_sum_rel + rels[remain, :]) - sqrt(curr_sum_rel)
            # sum over aspects.
            # But careful with memory on very large sizes.
            rels_remain = self.rels[remain_idxs, :]  # (R, Na)
            # sqrt difference:
            new_sqrts = np.sqrt(curr_sum_rel[np.newaxis, :] + rels_remain)
            old_sqrts = np.sqrt(curr_sum_rel[np.newaxis, :])
            asp_gain_arr = np.sum(new_sqrts - old_sqrts, axis=1)  # (R,)
            # aspeectCover
            asp_gain_arr = np.sum(self.aspect_weights[np.newaxis, :] * (new_sqrts - old_sqrts), axis=1)

            # total gain = Q_vals[remain] + lambda * asp_gain_arr
            total_gain_arr = self.Q_vals[remain_idxs] + self.lambda_ * asp_gain_arr

            # pick best
            ri = np.argmax(total_gain_arr)
            best_idx = remain_idxs[ri]
            best_gain = float(total_gain_arr[ri])

            if best_gain <= 0 and step == 0:
                pass

            marginal_gain_diff  = best_gain if prev_gain is None else best_gain - prev_gain
            if prev_gain is not None:
                print('prev_marginal_gain',)
                print('marginal_gain',marginal_gain_diff )

                if  abs(marginal_gain_diff)  < prev_gain * self.drop_ratio:
                    print(f"Stopping at step {step} due to marginal gain drop.")
                    break
            prev_gain = best_gain

            selected_mask[best_idx] = True
            selected.append(best_idx)
            curr_score += best_gain
            curr_sum_rel += self.rels[best_idx, :]


            history.append({
                "step": step + 1,
                "chosen_pair_idx": best_idx,
                "pair": self.pairs[best_idx],
                "gain": best_gain,
                "curr_score": curr_score,
                "curr_sum_rel": curr_sum_rel.copy().tolist(),
            })

        # convert selected indices to (a_text, b_text) and values
        selected_pairs = []
        for idx in selected:
            most_related_aspect_idx = np.argmax(self.rels[idx, :])
            most_related_aspect_name = self.aspects[most_related_aspect_idx]

            i_a, j_b = self.pairs[idx]
            selected_pairs.append({
                "pair_idx": idx,
                "a_idx": i_a,
                "b_idx": j_b,
                "a_text": self.Dm[i_a],
                "b_text": self.Dv[j_b],
                "aspect": most_related_aspect_name,
                "Q": float(self.Q_vals[idx]),
                "rels": self.rels[idx, :].tolist()
            })

        result = {
            "selected_pairs": selected_pairs,
            "selected_pair_indices": selected,
            "final_score": curr_score,
            "history": history,
            "aspect_list": self.aspects,
            "I_texts": self.I_texts,
            "o_m":self.o_m,
            "o_v":self.o_v,
            "aspect_weight": self.aspect_weights.tolist()
        }
        return result

# ---------------------------
# ----  ----
# ---------------------------
def convert(o):
    if isinstance(o, np.integer):
        return int(o)
    elif isinstance(o, np.floating):
        return float(o)
    elif isinstance(o, np.ndarray):
        return o.tolist()
    else:
        raise TypeError


if __name__ == "__main__":


    modelname = ""
    client_socket_sum = modelname
    # content_obj = open('summary_result/0504_CO_0_summary_results.txt', 'r', encoding="utf-8")
    # content_obj = open('summary_result/0504_EO_0_summary_results.txt', 'r', encoding="utf-8")
    path ="" # Summary results containing the majority opinion, divergent opinions, and their corresponding supporting evidence sets.
    content_obj = open(path, 'r', encoding="utf-8") #

    contents = content_obj.readlines()
    flag = ""

    try:
        for index, content in enumerate(contents):

            # 63
            # if index >=99:
            if index >=0:
                info = ast.literal_eval(content)
                topic = info['topic']
                o_m = info['major_opinion']
                o_v = info['divergent_opinion']
                Dm = info['sup_list']
                Dv = info['opp_list']
                docs = info['documents']

                #
                documents = [preprocess(tweet).strip() for tweet in docs]

                if len(Dv) == 0:
                    info['is_ture'] = 0  #
                    with open('explan_results/' + str(flag) + 'final_results' + '.txt', 'a+',
                              encoding='utf-8') as file:
                        # # 写入内容
                        file.write(str(info).replace('\n', '') + '\n')
                    continue
                else:
                    info['is_ture'] = 1

                with open('prompt/get_aspect_prompt.txt', 'r', encoding='utf-8') as file:
                    cointent = file.readlines()
                    get_aspect_prompt ='\n'.join(cointent)

                # minimal example
                # Dm = [
                #     "The company increased prices this year due to inflation.",
                #     "Service was poor and staff were rude in many instances.",
                #     "The new model has battery issues."
                # ]
                # Dv = [
                #     "Prices remained stable and customers are happy with value.",
                #     "Staff were friendly and helpful throughout our stay.",
                #     "Battery life improved in the latest update."
                # ]
                # o_m = "Prices are high and service is poor."
                # o_v = "Prices are reasonable and service is good."
                # topic = 'the review of a company'
                get_aspect_prompt = get_aspect_prompt.replace('{topic}',topic).replace('{document}',str(Dm+Dv))

                selector = EvidencePairSelector(
                    Dm=Dm,
                    Dv=Dv,
                    o_m=o_m,
                    o_v=o_v,
                    client_socket_sum=client_socket_sum,
                    get_aspect_prompt=get_aspect_prompt,
                    lambda_=2.0,
                    alpha=0.6,
                    m=3,
                    embedding_model="../pretrainmodels/all-MiniLM-L6-v2"
                )

                res = selector.greedy_select()
                res['documents'] = documents
                res['topic'] = topic

                print("Selected pairs (a_idx, b_idx) and short info:")
                for p in res["selected_pairs"]:
                    print(f" pair_idx={p['pair_idx']} (a{p['a_idx']}, b{p['b_idx']}), Q={p['Q']:.4f}, rels={p['rels']}")
                print("Final score:", res["final_score"])
                print("Aspects:", res["aspect_list"])
                print("aspect_weight:", res["aspect_weight"])
                with open('explan_results/' + str(flag) + 'final_results' + '.txt', 'a+',
                          encoding='utf-8') as file:
                    # json.dump(res, file, ensure_ascii=False,default=convert)
                    # file.write("\n")

                    file.write(str(res).replace('\n', '') + '\n')
    except Exception as ex:
        import traceback
        traceback.print_exc()
