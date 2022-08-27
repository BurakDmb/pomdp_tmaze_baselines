from multiprocessing import Process, Manager
import torch
import numpy as np
from queue import Empty


# Generates queues for each experiment and returns the list
# Each element corresponds to the connection element
def generateCommDict(start_dict, env_n_proc):
    comm_dict = {}
    for key in start_dict:
        if start_dict[key]:
            manager = Manager()
            # comm_list = manager.dict()
            # comm_list = manager.list()
            comm_list = list()
            for i in range(env_n_proc):
                produceQueue = manager.Queue()
                consumeQueue = manager.Queue()
                comm_list.append((produceQueue, consumeQueue))

            comm_dict[key] = comm_list
        else:
            comm_dict[key] = None
    return comm_dict


# This consumer constantly consumes queues to infer autoencoder model
# comm_dict_ includes each experiment
# comm_list includes a list with env_n_proc elements,
# each element corresponds to:
# 0th index: producer queue, 1st index: consumer queue
def ae_consumer(comm_dict_, ae_path, device):

    # Init AE model
    if ae_path is not None:

        ae_model_ = torch.load(ae_path).to(device)
        ae_model_ = ae_model_.module
        ae_model_.eval()
        ae_model = ae_model_

    data_dict = {}
    env_n_proc = {}
    for comm_key in comm_dict_:
        if comm_dict_[comm_key] is None:
            env_n_proc[comm_key] = 0
        else:
            env_n_proc[comm_key] = len(comm_dict_[comm_key])
        data_dict[comm_key] = np.zeros((env_n_proc[comm_key], 3, 48, 48))

    break_loop = False
    while not break_loop:
        for comm_key in comm_dict_:
            comm_list = comm_dict_[comm_key]
            if (comm_list is not None):
                # comm = list(comm_list)
                validRequests = [False]*len(comm_list)
                verr_count1 = 0
                for i in range(len(comm_list)):
                    try:
                        data_ = comm_list[i][0].get_nowait()
                        data_dict[comm_key][i:i+1, :] = data_
                        validRequests[i] = True
                    except Empty:
                        pass
                    except ValueError:
                        verr_count1 += 1
                if verr_count1 >= len(comm_list):
                    break_loop = True
                    break

                with torch.no_grad():
                    data = torch.tensor(
                        data_dict[comm_key],
                        requires_grad=False, device=device,
                        dtype=torch.float32)

                    obs, latent = ae_model(data)
                obs = obs.cpu().numpy()
                latent = latent.cpu().numpy()

                verr_count2 = 0
                for i in range(len(comm_list)):
                    if validRequests[i]:
                        try:
                            comm_list[i][1].put(
                                (obs[i: i+1, :], latent[i:i+1, :]))
                        except ValueError:
                            verr_count2 += 1

                if verr_count2 >= len(comm_list):
                    break_loop = True
                    break


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
