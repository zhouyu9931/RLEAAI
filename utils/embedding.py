import esm
import torch
from Bio import SeqIO

def esm_embedding(seq_dict):

    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    batch_converter = alphabet.get_batch_converter()
    model.eval()  # disables dropout for deterministic results
    embeddings = []
    for identifier, sequence in seq_dict.items():
        seq = [(identifier, sequence)]
        batch_labels, batch_strs, batch_tokens = batch_converter(seq)

        # Extract per-residue representations (on CPU)
        with torch.no_grad():
            results = model(batch_tokens, repr_layers=[33], return_contacts=True)
        representations = results["representations"][33]
        embeddings.append(representations)

    embeddings = [e[:, 1:-1] for e in embeddings]
    emb = torch.cat(embeddings, dim=1)

    return emb