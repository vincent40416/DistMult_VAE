# DistMult Experiment
The implementation of DistMult_VAE models is a fork of [PASSLEAF](https://github.com/Franklyncc/PASSLEAF); the implementation of deterministic knowledge graph embedding models is a fork of [OpenKE](https://github.com/thunlp/OpenKE).

## Environment
* python3 (>=3.6)
* tensorflow-gpu (==1.14)

For other dependencies, please see ```requirements.txt```. (Some of them may be UNNECESSARY!)


## Datasets
Full datasets for VAE models:
* ppi5k_no_psl
* nl27k_no_psl
* cn15k_no_psl

Filtered datasets for deterministic KG embedding models:
* ppi5k_openke_threshold_[threshold]_gen
* nl27k_openke_threshold_[threshold]_gen
* cn15k_openke_threshold_[threshold]_gen

[threshold] is the threshold for positive samples. The value may be one of the following: {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9}.

The datasets locate in the ```data``` folder.

## Models
In the following script examples, we use a batch size of 512, embedding dimension of 512, and save the checkpoint every 40 epochs. 

### VAE Model **with** pool-based semi-supervised learning
* DistMult_VAE: Uncertain ComplEx

Scripts: 
* Train:

    ```
    bash.sh
    ```

* Test:
    
    ```
    test.sh
    ```

＊ the model saving path name varies according to the current date.




