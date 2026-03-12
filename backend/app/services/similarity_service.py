from sentence_transformers import SentenceTransformer, util

# load model once
model = SentenceTransformer("all-MiniLM-L6-v2")


def calculate_similarity(claim_text: str, evidence_text: str):

    claim_embedding = model.encode(claim_text, convert_to_tensor=True)
    evidence_embedding = model.encode(evidence_text, convert_to_tensor=True)

    similarity = util.cos_sim(claim_embedding, evidence_embedding)

    return float(similarity[0][0])