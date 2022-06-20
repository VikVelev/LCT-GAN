# LCT-GAN

Improving tabular data synthesis, by introducing a novel latent gan architecture, using autoencoder as an embedding for tabular data and decreasing training time and use of computational resources.

# How to reproduce

One needs python 3.10 and poetry.

```
git clone https://github.com/VikVelev/LCT-GAN
cd LCT-GAN
poetry install
poetry shell
python main.py
```

All experiments, as outlined in the paper will run.

# Experiments
Experiments below are not relevant to the final results discussed in the paper.
They could be used as guidance to hyperparameter tuning.
## AE Experiments Results

```
### Time: 37:29 (AutoEncoder Only) 2.00s per epoch
bottleneck = 64, ae_epochs = 1000, ae_batch_size = 512
###

Average WD (Continuous Columns)  Average JSD (Categorical Columns)  Correlation Distance
                       0.027964                           0.159334               0.73152
Computing Machine Learning performance
          Acc       AUC  F1_Score
lr   1.207903  0.026464  0.011863
dt   3.378033  0.010379  0.044399
rf   5.619818  0.060401  0.065158
mlp  4.944211  0.041212  0.020813


### Time: 20:59 (AutoEncoder Only) 2.00s per epoch
bottleneck = 64, ae_epochs = 500, ae_batch_size = 512
###

Average WD (Continuous Columns)  Average JSD (Categorical Columns) Correlation Distance 
                       0.032151                           0.117791              0.433162
Computing Machine Learning performance                                                                                                                                                       Acc       AUC  F1_Score                                           
lr   0.071655  0.008918  0.041582                                                                                                                      
dt   2.108711  0.000290  0.039733
rf   4.821374  0.057608  0.081647
mlp  3.828437  0.032753  0.105722
```

## GAN Experiments Results
```
gan_epochs < 100 is very low, and it does not learn anything

### 7:08 (GAN Only) 2.00s per epoch
bottleneck=64, ae_epochs=300, gan_latent_dim=16, gan_epochs=200, batch_size for both = 512, n_critic = 5, with conditional vectors
###

Average WD (Continuous Columns)  Average JSD (Categorical Columns)  Correlation Distance                                                                                
                        0.04803                           0.323736              2.450655 

### 19:29 (GAN Only) 2.30s per epoch
bottleneck=64, ae_epochs=300, gan_latent_dim=16, gan_epochs=500, batch_size for both = 512, n_critic = 5, with conditional vectors
###

Average WD (Continuous Columns)  Average JSD (Categorical Columns)  Correlation Distance                                                                                
              0.036122                           0.310798              1.892707
Computing Machine Learning performance
          Acc       AUC  F1_Score
lr   4.360733  0.103154  0.219460
dt   5.056812  0.212945  0.236295
rf   9.356127  0.200241  0.307879
mlp  7.513563  0.154108  0.313866

### 45:31 (GAN Only) 2.60s per epoch
bottleneck=64, ae_epochs=300, gan_latent_dim=16, gan_epochs=1000, batch_size for both = 512, n_critic = 5, with conditional vectors
###

Average WD (Continuous Columns)  Average JSD (Categorical Columns)  Correlation Distance
                        0.03216                           0.293601              2.403066
Computing Machine Learning performance
           Acc       AUC  F1_Score
lr    5.476507  0.162489  0.199235
dt    7.114341  0.156668  0.158757
rf   10.410482  0.268033  0.357350
mlp  11.884533  0.263604  0.285133

###  Time: 17:26 (GAN Only) 3.00s per epoch
bottleneck=64, ae_epochs=300, gan_latent_dim=8, gan_epochs=350, batch_size for both = 256, n_critic = 5, with conditional vectors
###

Average WD (Continuous Columns)  Average JSD (Categorical Columns)  Correlation Distance
                        0.03662                           0.347303              2.424357
Computing Machine Learning performance
           Acc       AUC  F1_Score
lr    4.391442  0.158914  0.219560
dt   12.427065  0.245534  0.254995
rf   10.308118  0.306682  0.350255
mlp   7.800184  0.240715  0.310236


### Time: 21:55 (GAN Only) 4.00s per epoch (computer in use)
bottleneck=64, ae_epochs=1000, gan_latent_dim=16, gan_epochs=400, batch_size for both = 512, n_critic = 2, with conditional vectors
###
Average WD (Continuous Columns)  Average JSD (Categorical Columns)  Correlation Distance
                        0.040375                           0.275343              1.945418
Computing Machine Learning performance
           Acc       AUC  F1_Score
lr    8.803358  0.081050 -0.022428
dt    7.984441  0.079278  0.099133
rf   11.546729  0.145226  0.133768
mlp   8.209643  0.156457  0.101384

### Time: 21:44 (GAN Only) 2.00s per epoch
bottleneck=64, ae_epochs=1000, ae_batch_size=512, gan_latent_dim=16, gan_epochs=600, gan_batch_size = 100000, n_critic = 5, NO conditional vectors
###
   Average WD (Continuous Columns)  Average JSD (Categorical Columns)  Correlation Distance                                                             
0                         0.055747                           0.503827              4.664339   

Machine Learning performance isn't computable, as variety in generated data is non-existent

### Time 24:17 (GAN Only) 2.90s per epoch
bottleneck=64, ae_epochs=600, ae_batch_size=512, gan_latent_dim=16, gan_epochs=500, gan_batch_size = 512, n_critic = 2, with conditional vectors
###

Average WD (Continuous Columns)  Average JSD (Categorical Columns)  Correlation Distance
                       0.054021                           0.310133              3.348186
Computing Machine Learning performance
           Acc       AUC  F1_Score
lr    7.329307  0.172719  0.128875
dt   33.442522  0.342314  0.335950
rf   15.487767  0.501091  0.360692
mlp  15.375166  0.297836  0.244357

### Time 55:04 (GAN Only) 3.3s per epoch
bottleneck=64, ae_epochs=600, ae_batch_size=512, gan_latent_dim=32, gan_epochs=1000, gan_batch_size = 256, n_critic = 3, with conditional vectors
###

Average WD (Continuous Columns)  Average JSD (Categorical Columns)  Correlation Distance
                       0.052771                           0.348916              2.862465
Computing Machine Learning performance
           Acc       AUC  F1_Score
lr   18.353977  0.195117  0.122586
dt   14.904289  0.157145  0.171398
rf   16.255502  0.150762  0.137829
mlp  22.735183  0.214634  0.169580


###
bottleneck=64, ae_epochs=1000, ae_batch_size=512, gan_latent_dim=16, gan_epochs=700, gan_batch_size = 512, n_critic = 2, with conditional vectors
###

Computing statistical similarities
   Average WD (Continuous Columns)  Average JSD (Categorical Columns)  Correlation Distance
0                          0.04063                           0.266608               1.84446
Computing Machine Learning performance
           Acc       AUC  F1_Score
lr    6.305661  0.104329  0.130699
dt    5.384379  0.153486  0.186293
rf   10.134098  0.186097  0.268143
mlp  10.574266  0.188727  0.211743

```