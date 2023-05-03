#!/usr/bin/env python

#This script calculates log-likehood of a given model
import torch
import sys
import esm
import numpy as np
import argparse
from Bio import SeqIO
import tqdm
def calculate_likelihood(seq_arr,model,alphabet,device="cuda:0",batch_size=16):
    #calculate log-likelihood of a given model
    #seq_arr is an array of sequences
    #return log-likelihood
    arr = []
    batch_converter = alphabet.get_batch_converter()
    model = model.eval()
    model.to(device)
    for s in seq_arr:
        arr.append((s.id,str(s.seq).replace('*','')))
    batch_labels, batch_strs, batch_tokens = batch_converter(arr)
    batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

    final = []
    with torch.no_grad():
        for b in tqdm.tqdm(range(0,batch_tokens.shape[0],batch_size)):
            k = min(b+batch_size, batch_tokens.shape[0])
            batch = batch_tokens[b:k,:]
            results = model(batch.to(device), repr_layers=[34], return_contacts=False)

            probs = results["logits"].log_softmax(-1)
            one_hot = torch.nn.functional.one_hot(batch, num_classes=probs.shape[-1]).to(torch.float32).to(device)
            one_hot[:,:,1] = 0

            probs = probs * one_hot
            prob_sum = probs.sum(axis=-1)
            mean_prob = prob_sum.sum(axis=-1) / batch_lens[b:k].to(device)
            final.append(mean_prob)
    return torch.cat(final)




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate log-likelihood of a given model')
    parser.add_argument('-i', '--input', help='Input file', required=True)
    parser.add_argument('-m', '--model', help='Model', default="esm2_t33_650M_UR50D")
    parser.add_argument('--device', help='Device', default="cuda:0")
    parser.add_argument('--output','-o', help='Output file', default="out.csv")
    parser.add_argument('--batch_size', help='Batch size', default=16, type=int)
    args = parser.parse_args()
    model,alphabet = esm.pretrained.load_model_and_alphabet(args.model)
    seq_arr = list(SeqIO.parse(args.input,"fasta"))
    ll = calculate_likelihood(seq_arr,model,alphabet,args.device,args.batch_size)
    ll = ll.cpu().numpy()
    #save log-likelihood scores together with the sequence ids as csv file:
    ids = [s.id for s in seq_arr]
    np.savetxt(args.output,np.array([ids,ll],dtype='object').T,delimiter=",",fmt="%s")


            
