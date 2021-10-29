import torch
from botorch.models import SingleTaskGP
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels import ScaleKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
import numpy as np

from botorch.models import FixedNoiseGP
from gpytorch.priors import SmoothedBoxPrior
from gpytorch.kernels.kernel import Kernel
from gpytorch.kernels import ScaleKernel, CylindricalKernel, MaternKernel
from gpytorch.mlls import ExactMarginalLogLikelihood
from botorch import fit_gpytorch_model
from botorch.acquisition.analytic import ExpectedImprovement
from mesmo_botorch import MOMaxValueEntropy
from evaluate_policy import function_evaluate_policy
import time
import cma
import logging 
import sys
def configure_logger():
    global logger
    # warnings.filterwarnings("ignore")
    # logger_format = "%(levelname)s [%(asctime)s]: %(message)s"
    logger_level = logging.DEBUG #if args.debug else logging.ERROR
    logging.basicConfig(filename='bo_policy_search_500.log',
                            level=logger_level,
                            filemode='w')  # use filemode='a' for APPEND



NUM = 660#?

def initialize_model(train_x, train_y, covar_module, state_dict=None):
    # define models for objective and constraint
    model = SingleTaskGP(train_x, train_y, covar_module=covar_module).to(train_x)
    mll = ExactMarginalLogLikelihood(model.likelihood, model)
    # load state dict if it is passed
    if state_dict is not None:
        model.load_state_dict(state_dict)
    return mll, model


def evaluate_input(x):
    x = (x*torch.sqrt(torch.tensor(NUM_FL + NUM_FB + NUM_NB, dtype=torch.float))).unsqueeze(0).numpy()
    x = (-1 + 2*(np.max(x)-x)/(np.max(x)-np.min(x)))
    regressor_vfi1 = x[:, :NUM]
    regressor_vfi2 = x[:, NUM: 2*NUM]
    regressor_vfi3 = x[:, 2*NUM : 3*NUM]
    regressor_vfi4 = x[:, 3*NUM : 4*NUM]
	#['fft', 'canneal', 'dedup', 'fluid', 'lu', 'radix', 'vips', 'water']
    # Function takes the three policies as inputs and outputs the energy, exe time
    wl_energy, wl_exe, wl_ppw = function_evaluate_policy(regressor_vfi1, regressor_vfi2, regressor_vfi3, regressor_vfi4, ['fft'])

    print(f"{np.mean(wl_energy), np.mean(wl_exe)}")
    #return (-1*np.mean(wl_energy), np.mean(wl_ppw), wl_energy, wl_ppw)
    return (-1*np.mean(wl_exe), np.mean(wl_ppw), wl_exe, wl_ppw) # why not energy here ? energy edp or ppw ?

def main():
    # initial set of random inputs 
    INIT_SAMPLES = 20
    NUM_ITERS = 500
    covar_module = ScaleKernel(base_kernel = CylindricalKernel(num_angular_weights=3, radial_base_kernel=MaternKernel(), active_dims = torch.arange(NUM_FL)) 
                               + CylindricalKernel(num_angular_weights=3, radial_base_kernel=MaternKernel(), active_dims = torch.arange(NUM_FL, NUM_FL + NUM_FB)) 
                               + CylindricalKernel(num_angular_weights=3, radial_base_kernel=MaternKernel(), active_dims = torch.arange(NUM_FL + NUM_FB, NUM_FL + NUM_FB + NUM_NB)))    
    logger = logging.getLogger() 
    configure_logger()
    for nruns in range(25):
        train_x = torch.rand((INIT_SAMPLES, NUM_FL + NUM_FB + NUM_NB))
        train_x = train_x/torch.sqrt(torch.tensor(NUM_FL + NUM_FB + NUM_NB, dtype=torch.float))

        wl_energy_list, wl_exe_list = [], []
        # initializing surrogate model with INIT_SAMPLES points
        print(f"initalizing surrogate model...")
        outputs_f = evaluate_input(train_x[0])
        wl_energy_list.append(outputs_f[2])
        wl_exe_list.append(outputs_f[3])
        train_y_1 = torch.tensor([outputs_f[0]])
        train_y_2 = torch.tensor([outputs_f[1]])
        for i in range(1, INIT_SAMPLES):
            outputs_s = evaluate_input(train_x[i])
            wl_energy_list.append(outputs_s[2])
            wl_exe_list.append(outputs_s[3])
            train_y_1 = torch.cat([torch.tensor(outputs_s[0]).unsqueeze(0), train_y_1])
            train_y_2 = torch.cat([torch.tensor(outputs_s[1]).unsqueeze(0), train_y_2])
        train_y_1 = train_y_1.unsqueeze(-1)
        train_y_2 = train_y_2.unsqueeze(-1)
        print(f"Finished initialization!")

        mll_ei_first, model_ei_first = initialize_model(train_x, train_y_1, covar_module)
        mll_ei_sec, model_ei_sec = initialize_model(train_x, train_y_2, covar_module)
        for num_iters in range(INIT_SAMPLES, NUM_ITERS):
            start_time = time.time()
            fit_gpytorch_model(mll_ei_first)
            fit_gpytorch_model(mll_ei_sec)
            print(f"Fitting time {time.time() - start_time:.2f}")
            logger.debug("Fitting time: %s"%(time.time() - start_time))
            MES = MOMaxValueEntropy([model_ei_first, model_ei_sec], 2)
            # AFO procedure
            start_time = time.time()
            x0 = torch.rand(NUM_FL + NUM_FB + NUM_NB).numpy()
            es = cma.CMAEvolutionStrategy(
                x0=x0,
                sigma0=0.2,
                inopts={'bounds': [0, 1], "popsize": 50},
            )
            # speed up things by telling pytorch not to generate a compute graph in the background
            with torch.no_grad():
                i = 0
                while not es.stop():
                    xs = es.ask()  # as for new points to evaluate
                    # convert to Tensor for evaluating the acquisition function
                    X = torch.tensor(xs, dtype=torch.float)
                    # evaluate the acquisition function (optimizer assumes we're minimizing)
                    Y = -1 * MES.run(X/torch.sqrt(torch.tensor(NUM_FL + NUM_FB + NUM_NB, dtype=torch.float)))  # acquisition functions require an explicit q-batch dimension
                    y = Y.view(-1).numpy()  # convert result to numpy array
                    es.tell(xs, y)  # return the result to the optimizer
                    print(f"current best : {es.best.f}")
                    i += 1
                    if (i > 10):
                        break
            print(f"AFO time {time.time() - start_time:.2f}")
            logger.debug("AFO time: %s"%(time.time() - start_time))
            # convert result back to a torch tensor
            best_next_input = torch.from_numpy(es.best.x)
            print(f"best_next_input: {best_next_input.size()}")
            train_x = torch.cat([train_x, best_next_input.unsqueeze(0)/torch.sqrt(torch.tensor(NUM_FL + NUM_FB + NUM_NB, dtype=torch.float))])
            print(f"train_x size: {train_x.size()}")
            print(f"Iteration {num_iters}:")
            #print(f"{idx}th point selected ", end='')
            outputs = evaluate_input(train_x[-1])
            wl_energy_list.append(outputs[2])
            wl_exe_list.append(outputs[3])
            train_y_1 = torch.cat([train_y_1, torch.tensor(outputs[0]).reshape(1, 1)])
            train_y_2 = torch.cat([train_y_2, torch.tensor(outputs[1]).reshape(1, 1)])
            print(f"with value: {(train_y_1[-1].item(), train_y_2[-1].item())}")
            mll_ei_first, model_ei_first = initialize_model(train_x, train_y_1, covar_module)
            mll_ei_sec, model_ei_sec = initialize_model(train_x, train_y_2, covar_module)
            torch.save({'inputs': train_x, 'outputs_1': train_y_1, 'outputs_2': train_y_2, 'wl_energy_list':wl_energy_list, 'wl_exe_list':wl_exe_list}, 'wl_exe_ppw_global_mobo_data_'+sys.argv[1]+'app_min'+str(nruns)+'.pkl')



if __name__ == '__main__':
    main()
