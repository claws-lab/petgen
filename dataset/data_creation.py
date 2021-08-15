#%%
# read the pkl files and generate the correponding text data
#%%
import pickle as pkl
#%%
dir = None # To be specified
#%%
sents2label = pkl.load(open(f"{dir}/Seq2label.pkl", "rb"))
#%%
sents2context = pkl.load(open(f"{dir}/Seq2context.pkl", "rb"))
#%%
#%%
with open("yelp.txt", "w") as f:
    with open("yelp_text.txt", "w") as f_text:
        for sents in sents2label:
            #%%
            for sent in sents[:-1]:
                f.write(sent+"\n")
            f_text.write(sents[-1]+"\n")
#%%
with open("yelp_context.txt", "w") as f:
    for contexts in sents2context.values():
        for context in contexts:
            f.write(context+"\n")
#%%


