# Multi producer single consumer yapisi
# Her producer icin ayri bir pipe olustur.
# Loopda donecek olan kodda sirayla her pipe'da veri var mi bak.
# Eger veri varsa

# from time import sleep
# from random import random
from multiprocessing import Process
from multiprocessing import Manager
import torch
import numpy as np
import torch2trt


# Generates queues for each experiment and returns the list
# Each element corresponds to the connection element
# TODO generate all the keys in here
def generateCommDict(start_dict, env_n_proc):
    comm_dict = {}
    for key in start_dict:
        if start_dict[key]:
            manager = Manager()
            # comm_variable = manager.dict()
            comm_variable = manager.list()
            for i in range(env_n_proc):
                comm_variable.append(manager.list(
                    [False, False, None, None, None]))
            # comm_lock = manager.Lock()
            # comm_variable['request'] = [False]*env_n_proc
            # comm_variable['request_completed'] = [False]*env_n_proc
            # comm_variable['data'] = [None]*env_n_proc
            # comm_variable['result_obs'] = [None]*env_n_proc
            # comm_variable['result_latent'] = [None]*env_n_proc
            # comm_variable['lock'] = comm_lock
            comm_dict[key] = comm_variable
        else:
            comm_dict[key] = None
    return comm_dict


# comm variable i vektorel olacak, consumer ise her
# zaman loop icerisinde donerek batch şeklinde tum
# inputlarin output'ini hesaplayacak.
# Eger o an request eden yok ise onceden initialize
# edilmis bos zero tensoru kullanilacak.
# Comm variable:
# 0 request
# 1 request_completed
# 2 data
# 3 result_obs
# 4 result latent
def ae_consumer(comm_list_, ae_path, device):

    # Init AE model
    if ae_path is not None:

        ae_model_ = torch.load(ae_path).to(device)
        ae_model_ = ae_model_.module.to(device)
        ae_model_.eval()

        ae_model = tortch2trt.TRTModule()
        ae_model.load_state_dict(torch.load())

        # ae_model = torch2trt.torch2trt(
        #     ae_model_, [torch.tensor(
        #         np.zeros((1, 3, 48, 48)),
        #         dtype=torch.float32).cuda()],
        #     strict_type_constraints=True,
        #     # fp16_mode=False,
        #     use_onnx=True
        #     )
        # ae_model = TRTModule(ae_model_)
        # ae_model.load_state_dict(ae_model_)

    data_dict = {}
    env_n_proc = {}
    for comm_key in comm_list_:
        if comm_list_[comm_key] is None:
            env_n_proc[comm_key] = 0
        else:
            env_n_proc[comm_key] = len(comm_list_[comm_key])
        data_dict[comm_key] = np.zeros((env_n_proc[comm_key], 3, 48, 48))

    while True:
        # comm_list = dict(comm_list_)
        for comm_key in comm_list_:
            comm = comm_list_[comm_key]
            if (comm is not None):
                for i in range(len(comm)):
                    if comm[i][0]:
                        data_dict[comm_key][i:i+1, :] = comm[i][2]

                with torch.no_grad():
                    data = torch.tensor(
                        data_dict[comm_key],
                        requires_grad=False, device=device,
                        dtype=torch.float32)

                    obs, latent = ae_model(data)

                for i in range(len(comm)):

                    comm[i][3] = obs[i: i+1, :].cpu().numpy()
                    comm[i][4] = latent[i:i+1, :].cpu().numpy()
                    if comm[i][0]:
                        comm[i][0] = False
                        comm[i][1] = True


if __name__ == '__main__':

    device = 'cuda:0'
    ae_path = "models/ae.torch"
    start_dict = {
        'ae_no_mem': True,
        'ae_smm_lastk': True,
        'ae_smm_bk': True,
        'ae_smm_ok': True,
        'ae_smm_ok_intr': True,
        'ae_smm_oak': True,
        'ae_smm_oak_intr': True,
        'ae_lstm': True,
        'cnn_no_mem': True,
    }

    comm_list = generateCommDict(start_dict)

    ae_consumer_process = Process(
        target=ae_consumer, args=(comm_list, ae_path, device))
    ae_consumer_process.start()

    # ae_consumer_process.terminate()
