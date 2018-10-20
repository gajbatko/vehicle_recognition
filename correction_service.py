from correction import correct


#reading data
def read(filename):
    list = [[] for i in range(7)]
    with open('\\'.join(['D:\\mgr\\busy_all', filename])) as f:
        for line in f:
            for x in range(7):
                list[x].append(float(line.split()[x]))
    return list



#phase correction
def correct_phase(data_r_1, data_x_1, data_r_2, data_x_2):

    sensor1_data = [correct(r, x, -29) for r, x in zip(data_r_1, data_x_1)]
    sensor2_data = [correct(r, x, -32.5) for r, x in zip(data_r_2, data_x_2)]

    return sensor1_data, sensor2_data


def get_data_from_sensors(filename):

    data_list = read(filename)
    sensor1_data, sensor2_data = correct_phase(data_list[1], data_list[2], data_list[5], data_list[6])
    data_list.append([i[0] for i in sensor1_data])
    data_list.append([i[1] for i in sensor1_data])
    data_list.append([i[0] for i in sensor2_data])
    data_list.append([i[1] for i in sensor2_data])

    return data_list[7][:], data_list[8][:], data_list[3][:], data_list[4][:], data_list[9][:], data_list[10][:], data_list[0][:]



