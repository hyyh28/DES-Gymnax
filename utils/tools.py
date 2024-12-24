def get_customers_changing(obs, workers):
    customers_changing_dict = {worker: {"time": [], "num": [], "average_queue_length": 0} for worker in range(workers)}
    for key in customers_changing_dict:
        customers_changing_list = obs[key][:, 0].tolist()
        customers_changing_time_list = obs[key][:, 1].tolist()
        customers_changing_dict[key]["time"] = customers_changing_time_list
        customers_changing_dict[key]["num"] = customers_changing_list
        time_skip_list = [customers_changing_time_list[i+1] - customers_changing_time_list[i] for i in range(len(customers_changing_time_list)-1)]
        average_queue_length_list = [time_skip_list[i] * customers_changing_list[i] for i in range(len(customers_changing_time_list)-1)]
        average_queue_length = sum(average_queue_length_list) / customers_changing_time_list[-1]
        customers_changing_dict[key]["average_queue_length"] = average_queue_length
    return customers_changing_dict

def get_average_waiting_time(obs, workers):
    average_waiting_time_dict = {worker: {"time": [], "num": []} for worker in range(workers)}
    for key in average_waiting_time_dict:
        served_customers_num_list = obs[key][:, 2].tolist()
        # customers_changing_time_list = obs[key][:, 1].tolist()
        customers_total_waiting_time_list = obs[key][:, 3].tolist()
        average_waiting_time_dict[key]["time"] = customers_total_waiting_time_list
        average_waiting_time_dict[key]["num"] = served_customers_num_list
    average_waiting_time = {worker:average_waiting_time_dict[worker]["time"][-1] / average_waiting_time_dict[worker]["num"][-1] for worker in range(workers)}
    return average_waiting_time_dict, average_waiting_time


