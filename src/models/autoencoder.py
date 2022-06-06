import pandas as pd
from src.utils.transformer import DataTransformer
from src.utils.data_preparation import DataPrep
from src.utils.ctabgan_synthesizer import Sampler, Condvec

from tqdm import tqdm
import torch
import numpy as np
from torch.nn import functional as F
from torch import nn, optim
from .architectures import FCDecoder, FCEncoder

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor


class LatentTAE:

    """
    AutoEncoder auxiliary class using and training 
    an AutoEncoderModel to generate intermediary representation of a dataset

     - __init__(...) -> handles instantiating of the object with specified input parameters
     - fit(...) -> takes care of pre-processing and fits the AutoEncoder to the input data 
     - encode(tabular_data) -> returns a latent representation
     - decode(latent_data) -> tabular_data
     - get_latent_dataset() -> [latent]
    """

    def __init__(self,
                 embedding_size=16,
                 raw_csv_path="./data/Adult.csv",
                 test_ratio=0.20,
                 categorical_columns=['workclass', 'education', 'marital-status',
                                      'occupation', 'relationship', 'race', 'gender', 'native-country', 'income'],
                 log_columns=[],
                 mixed_columns={'capital-loss': [0.0], 'capital-gain': [0.0]},
                 integer_columns=['age', 'fnlwgt', 'capital-gain',
                                  'capital-loss', 'hours-per-week'],
                 problem_type={"Classification": 'income'}):

        self.__name__ = 'AutoEncoder'
        self.raw_df = pd.read_csv(raw_csv_path)
        self.ae = AutoEncoder(
            {"embedding_size": embedding_size, "log_interval": 5})
        self.test_ratio = test_ratio
        self.categorical_columns = categorical_columns
        self.log_columns = log_columns
        self.mixed_columns = mixed_columns
        self.integer_columns = integer_columns
        self.problem_type = problem_type
        self.train_data = None

    def fit(self, n_epochs=5000):

        self.data_prep = DataPrep(
            self.raw_df,
            self.categorical_columns,
            self.log_columns,
            self.mixed_columns,
            self.integer_columns,
            self.problem_type,
            self.test_ratio
        )

        self.transformer = DataTransformer(
            train_data=self.data_prep.df,
            categorical_list=self.data_prep.column_types["categorical"],
            mixed_dict=self.data_prep.column_types["mixed"]
        )

        self.transformer.fit()
        self.train_data = self.transformer.transform(self.data_prep.df.values)

        data_dim = self.transformer.output_dim
        data_info = self.transformer.output_info

        self.ae.train(self.train_data, data_dim, data_info, epochs=n_epochs)

        ##### TEST #####
        print("######## DEBUG ########")
        real = np.asarray([self.train_data[0]])

        latent = self.ae.encode(real)
        reconstructed = self.ae.decode(latent)

        inverse_real = self.transformer.inverse_transform(real)
        recon_inverse = self.transformer.inverse_transform(
            reconstructed.cpu().detach().numpy())

        table_real = self.data_prep.inverse_prep(inverse_real)
        table_recon = self.data_prep.inverse_prep(recon_inverse)

        print(table_real)
        print()
        print(table_recon)
        #### END OF TEST ####

    def encode(self, datum):
        real = np.asarray([datum])
        return self.ae.encode(real)

    def decode(self, latent, batch=False):
        if batch:
            table = []
            latent = latent.cpu().detach().numpy()

            for l in tqdm(latent):
                reconstructed = self.ae.decode(Tensor(l))
                reconstructed = np.asarray(
                    [reconstructed.cpu().detach().numpy()])

                recon_inverse = self.transformer.inverse_transform(
                    reconstructed)
                table_recon = self.data_prep.inverse_prep(recon_inverse)
                table.append(table_recon)

            return pd.concat(table)
        else:
            table = []
            for l in tqdm(latent):
                reconstructed = self.ae.decode(l)
                reconstructed = reconstructed.cpu().detach().numpy()

                recon_inverse = self.transformer.inverse_transform(
                    reconstructed)
                table_recon = self.data_prep.inverse_prep(recon_inverse)
                table.append(table_recon)

            return pd.concat(table)

    def get_latent_dataset(self, leave_pytorch_context=False):

        cond_generator = Condvec(self.train_data, self.transformer.output_info)

        latent_dataset = []
        for d in self.train_data:
            latent_dataset.append(self.encode(d).cpu().detach(
            ).numpy() if leave_pytorch_context else self.encode(d))

        return latent_dataset


class AENetwork(nn.Module):
    def __init__(self, args, input_dim):
        super(AENetwork, self).__init__()

        self.encoder = FCEncoder(args["embedding_size"], input_size=input_dim)
        self.decoder = FCDecoder(args["embedding_size"], input_size=input_dim)
        self.input_dim = input_dim

    def encode(self, x):
        return self.encoder(x)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x.view(-1, self.input_dim))
        return self.decode(z)


class AutoEncoder(object):

    def __init__(self, args):
        self.args = args  # has to have 'embedding_size' and 'cuda' = True
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.cond_generator = None

    def loss_function(self, recon_x, x, input_size):
        # BCE = F.binary_cross_entropy(recon_x, x.view(-1, input_size), reduction='sum')
        return F.mse_loss(recon_x, x.view(-1, input_size), reduction="sum")

    def encode(self, x):
        datum = torch.from_numpy(x).to(
            self.device).view(-1, self.input_size).float()
        return self.model.encode(datum)

    def decode(self, z):
        return self.model.decode(z)

    def train(self, data, output_dim: int, output_info, epochs=5000, batch_size=1024):

        # initializing the sampler object to execute training-by-sampling
        data_sampler = Sampler(data, output_info)
        # initializing the condvec object to sample conditional vectors during training
        cond_generator = Condvec(data, output_info)

        col_size_d = output_dim  # + cond_generator.n_opt
        self.input_size = col_size_d

        self.model = AENetwork(self.args, input_dim=col_size_d)
        self.model.to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-3)

        self.model.train()
        train_loss = 0

        last_loss = 0

        for i in tqdm(range(epochs)):

            # sample all conditional vectors for the training
            cond_vecs = cond_generator.sample_train(batch_size)
            c, m, col, opt = cond_vecs
            c = torch.from_numpy(c).to(self.device)
            m = torch.from_numpy(m).to(self.device)

            # sampling real data according to the conditional vectors and shuffling it before feeding to discriminator to isolate conditional loss on generator
            perm = np.arange(batch_size)
            np.random.shuffle(perm)
            real = data_sampler.sample(batch_size, col[perm], opt[perm])

            real = torch.from_numpy(real.astype('float32')).to(self.device)

            # storing shuffled ordering of the conditional vectors
            c_perm = c[perm]
            batch = torch.cat([real], dim=1).to(self.device)
            data = batch

            self.optimizer.zero_grad()
            recon_batch = self.model(data)
            loss = self.loss_function(
                recon_batch, data, input_size=self.input_size)
            loss.backward()
            train_loss += loss.item()
            self.optimizer.step()
            last_loss = (loss.item() / len(data))

            # print(loss.item() / len(data))

        print(last_loss)
