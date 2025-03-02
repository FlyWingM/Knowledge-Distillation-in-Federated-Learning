import argparse

def parse_args():

    """
    Parse command line arguments for server configuration, training parameters, 
    and model settings.
    """
    parser = argparse.ArgumentParser(description="Parse arguments for federated learning setup.")

    # Label extenstion
    label_extension_group = parser.add_argument_group('Server and Training Configuration')
    label_extension_group.add_argument('--server_flag', action='store_true', help='Enable further FL training in server')
    label_extension_group.add_argument('--ep_server', type=int, default=0, help='Epochs for further FL training in server')
    label_extension_group.add_argument('--label_f', type=bool, default=False, help="Enabler for label extension")
    label_extension_group.add_argument('--num_label_ELS', type=int, default=25, help='Number of lable set in non-IID dataset')
    label_extension_group.add_argument('--label_ex', type=int, default=1, help='Number of additional label set in marginal dataset')
    
    # Continuous Training
    cont_train_group = parser.add_argument_group('Continuous Training')
    cont_train_group.add_argument('--cont_train', type=bool, default=False, help='Enable continuous training using the previous model')
    cont_train_group.add_argument('--cont_train_rl', type=bool, default=False, help='Enable continuous training using the previous model')
    cont_train_group.add_argument('--cont_tr_iter', type=int, default=0, help='Last number of iterations for training the previous model')
    cont_train_group.add_argument('--cont_tr_acc', type=float, default=0.0, help='Highest accuracy of the previous model')
    
    # Dataset and Model Configuration ResNet9  densenet121
    data_model_group = parser.add_argument_group('Dataset and Model Configuration')
    data_model_group.add_argument('--data', type=str, default='cinic10', help="Name of dataset")
    data_model_group.add_argument('--nn', type=str, default=None, help='Neural network architecture')
    data_model_group.add_argument('--pre_trained', type=bool, default=False, help='Pre-trained neural network')
    data_model_group.add_argument('--nn_trainable', type=bool, default=False, help='Trainable pre-trained neural network')
    data_model_group.add_argument('--fl', type=str, default='FedAvg', help='FL algorithm')
    data_model_group.add_argument('--distr_type', type=str, default='Dirichlet', help='Data distribution: Dirichlet and Label-based')
    data_model_group.add_argument('--pre_distr', type=str, default='d0', help='Predefined data distribution')
    data_model_group.add_argument('--dirich', type=float, default=0.03, help='Dirichlet distribution')
    
    # Training and Distribution
    train_distribution_group = parser.add_argument_group('Training and Distribution')
    train_distribution_group.add_argument('--num_s', type=int, default=0, help="Number of the shared dataset")
    train_distribution_group.add_argument('--num_g_iter', type=int, default=900, help="Number of global iterations")
    train_distribution_group.add_argument('--l_iter_group', type=int, default=None, help="Number of local iterations")
    train_distribution_group.add_argument('--dirich_min', type=int, default=10, help="Dirichlet distribution minimum number")
    train_distribution_group.add_argument('--di_prop_n', type=str, default='1', help="Dirichlet distribution minimum number")
    train_distribution_group.add_argument('--di_gen_flag', type=bool, default=False, help="Dirichlet distribution minimum number")

    # Debug and Save
    debug_save_group = parser.add_argument_group('Debug flags and saving')
    debug_save_group.add_argument('--debug_s', type=bool, default=True, help="Debug for server training")

    # Client information
    client_info_group = parser.add_argument_group('Client information')
    client_info_group.add_argument('--num_clients', type=int, default=None, help="number of Clients")
    client_info_group.add_argument('--c_sel_cri', type=str, default=None, help="Client selection by criteria, acc, carbon, weight")
    client_info_group.add_argument('--lr_carbon', type=str, default=None, help="Client selection by Carbon in Inforcement Learning")
    client_info_group.add_argument('--dynamic_sel', type=str, default=None, help="Dynamic client selection by Carbon in Inforcement Learning")
    client_info_group.add_argument('--l_explore', type=str, default=None, help="Decision out of multi-selection per a global iteration")
    client_info_group.add_argument('--c_sel_weighting', type=str, default=None, help="weighting_exponential, weighting_inverse")
    client_info_group.add_argument('--c_sel_rate', type=float, default=None, help="Client selection by ratio")

    # Federated arguments
    fed_argu_group = parser.add_argument_group('Federated arguments')
    fed_argu_group.add_argument('--epochs', type=int, default=10, help="rounds of training")
    fed_argu_group.add_argument('--frac', type=float, default=0.1, help="the fraction of clients: C")
    fed_argu_group.add_argument('--local_ep', type=int, default=5, help="the number of local epochs: E")
    fed_argu_group.add_argument('--local_bs', type=int, default=10, help="local batch size: B")
    fed_argu_group.add_argument('--bs', type=int, default=128, help="test batch size")
    fed_argu_group.add_argument('--lr', type=float, default=0.01, help="learning rate")
    fed_argu_group.add_argument('--momentum', type=float, default=0.5, help="SGD momentum (default: 0.5)")
    fed_argu_group.add_argument('--split', type=str, default='user', help="train-test split type, user or sample")

    # model arguments
    model_argu_group = parser.add_argument_group('Model arguments')
    model_argu_group.add_argument('--kernel_num', type=int, default=9, help='number of each kind of kernel')
    model_argu_group.add_argument('--kernel_sizes', type=str, default='3,4,5',
                        help='comma-separated kernel size to use for convolution')
    model_argu_group.add_argument('--norm', type=str, default='batch_norm', help="batch_norm, layer_norm, or None")
    model_argu_group.add_argument('--num_filters', type=int, default=32, help="number of filters for conv nets")
    model_argu_group.add_argument('--max_pool', type=str, default='True',
                        help="Whether use max pooling rather than strided convolutions")


    # Environment arguments
    environ_argu_group = parser.add_argument_group('Environmental arguments')
    environ_argu_group.add_argument('--SBATCH', type=bool, default=False, help='Is a job submitted to sbatch ')


    args = parser.parse_args()
    return args