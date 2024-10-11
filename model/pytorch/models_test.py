import os
import time
import numpy as np
import torch
import pandas as pd
import yaml
import argparse
from torch.utils.tensorboard import SummaryWriter
from lib import utils, metrics
from model.pytorch.dcrnn_model import DCRNNModel
from model.pytorch.tsf_model import TSFModel, count_parameters
from model.pytorch.gtn_model import GTNModel
from model.pytorch.itgtn import iTGTNModel
from model.pytorch.itn import iTNModel
from model.pytorch.itpgtn import iTPGTNModel
from model.pytorch.itgcn import iTGCNModel
import shutil

class ModelsSupervisor:
    def __init__(self, models, pretrained_model_path, config_filename, cuda, **kwargs):
        self.device = torch.device(cuda if torch.cuda.is_available() else "cpu")
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')
        self._data_kwargs['seq_len'] = int(self._model_kwargs.get('seq_len'))
        self._data_kwargs['horizon'] = int(self._model_kwargs.get('horizon'))

        self._log_dir = self._get_log_dir(models, config_filename, kwargs)
        self._writer = SummaryWriter('runs/' + self._log_dir)
        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        self._data = utils.load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']

        graph_pkl_filename = self._data_kwargs.get('graph_pkl_filename')
        if self._data_kwargs.get('use_graph'):
            _, _, adj_mx = utils.load_graph_data(graph_pkl_filename)
            self.graph = adj_mx
        else:
            self.graph = None

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.horizon = int(self._model_kwargs.get('horizon', 1))

        model_list = {
            'dcrnn': DCRNNModel,
            'tsf': TSFModel,
            'gtn': GTNModel,
            'itn': iTNModel,
            'itgtn': iTGTNModel,
            'itgcn': iTGCNModel,
            'itpgtn': iTPGTNModel
        }
        init_model = model_list[models]
        self.amodel = init_model(self._logger, cuda, **self._model_kwargs).to(self.device)
        self._logger.info("config_filename:%s", config_filename)
        self._logger.info("device:%s", self.device)
        self._logger.info("Model created")

        self.load_pre_model(pretrained_model_path)

    @staticmethod
    def _get_log_dir(loadmodel, config_name, kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            seq_len = int(kwargs['model'].get('seq_len'))
            horizon = int(kwargs['model'].get('horizon'))
            run_id = '%s_%s_l_%d_h_%d_lr_%g_bs_%d/' % (
                time.strftime('%Y%m%d_%H%M%S'),
                loadmodel,
                seq_len, horizon,
                learning_rate, batch_size
            )
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, 'log', loadmodel, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        shutil.copy(config_name, log_dir)
        return log_dir

    def load_pre_model(self, model_path):
        self._setup_graph()
        checkpoint = torch.load(model_path, map_location='cpu')
        self.amodel.load_state_dict(checkpoint['model_state_dict'],strict=False)
        self._logger.info("Loaded model from {}".format(model_path))

    def _setup_graph(self):
        with torch.no_grad():
            self.amodel = self.amodel.eval()
            val_iterator = self._data['val_loader'].get_iterator()
            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                self.amodel(x, self.graph, batches_seen=0)
                break

    def evaluate(self, dataset='test'):
        with torch.no_grad():
            self.amodel = self.amodel.eval()
            data_loader = self._data['{}_loader'.format(dataset)].get_iterator()
            y_truths = []
            y_preds = []
            for idx, (x, y) in enumerate(data_loader):
                x, y = self._prepare_data(x, y)
                output = self.amodel(x, self.graph)
                
                y_truths.append(y.cpu().numpy())
                y_preds.append(output.cpu().numpy())
         

            y_truths = np.concatenate(y_truths, axis=1)
            y_preds = np.concatenate(y_preds, axis=1)

            y_truths_scaled = []
            y_preds_scaled = []
            for t in range(y_preds.shape[0]):
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)
             # 将预测和真实值reshape为(时间步, 节点数)的形状
            
            y_truths_scaled = np.array(y_truths_scaled).reshape(-1,207)
            y_preds_scaled = np.array(y_preds_scaled).reshape(-1,207)
        return y_preds_scaled, y_truths_scaled

    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        return x.to(self.device), y.to(self.device)

    def _get_x_y(self, x, y):
        x0 = x[..., 0]
        y0 = y[..., 0]
        x = torch.from_numpy(x0).float()
        y = torch.from_numpy(y0).float()

        
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        return x, y

def main(args):
    with open(args.config_filename, encoding='utf-8') as f:
        supervisor_config = yaml.load(f, Loader=yaml.FullLoader)
        print('config_filename:', args.config_filename)
        supervisor = ModelsSupervisor(args.models,  args.pretrained_model_dir, args.config_filename, args.cuda, **supervisor_config)

        predictions, truths = supervisor.evaluate(dataset='test')


        paired_data = {}
        for i in range(predictions.shape[1]):  # 遍历节点数
            paired_data[f'node_{i}_prediction'] = predictions[:, i]
            paired_data[f'node_{i}_truth'] = truths[:, i]

        # 创建 DataFrame
        df = pd.DataFrame(paired_data)

        # 去除空值
        df = df.dropna()
        df = df[(df != 0).all(axis=1)]

        # 保存到 CSV 文件
        output_file = os.path.join(supervisor._log_dir, 'model_output.csv')
        df.to_csv(output_file, index=False, header=True)

        print("模型输出已保存到:", output_file)




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--models', default=None, type=str, help='MODEL: TSF|DCRNN|LSTM|TGTN')
    parser.add_argument('--config_filename', default=None, type=str, help='Configuration filename for restoring the model.')
    parser.add_argument('--use_cpu_only', default=False, type=bool, help='Set to true to only use cpu.')
    parser.add_argument('--load_pretrained', default=True, type=bool, help='Whether to load a pretrained model.')
    parser.add_argument('--pretrained_model_dir', default=None, type=str, help='Directory of the pretrained model.')
    parser.add_argument('--cuda', default='cuda:0', type=str)
    args = parser.parse_args()
    main(args)
