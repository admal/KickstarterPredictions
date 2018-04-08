from data import load_data

if __name__ == '__main__':
    data = load_data()
    total_data = len(data)
    total_successes = len(data[data.state == 'successful'])
    total_fails = len(data[data.state == 'failed'])
    total_other = total_data - total_successes - total_fails

    s_rate = round(total_successes / total_data * 100, 2)
    f_rate = round(total_fails / total_data * 100, 2)
    o_rate = round(total_other / total_data * 100, 2)
    print("Total number of projects: {}".format(total_data))
    print("Successes: {} ({}%), fails: {} ({}%), other: {} ({}%)".format(total_successes, s_rate , total_fails, f_rate, total_other, o_rate))



