from dataclasses import dataclass
import math
import numpy as np
import matplotlib.pyplot as plt
# from simple_pid import PID
import csv


fig, ax = plt.subplots()


@dataclass
class PointMass:
    id: int
    x: float
    y: float


nb_ticks = 100
global curr_tick
nb_setpoints = 21
nb_agents = 6


# fst = PointMass(0, 2.0, 2.0)
# fst_fst = PointMass(1, 1.0, 1.0)
# fst_snd = PointMass(2, 3.0, 1.0)
# snd_fst = PointMass(3, 0.0, 0.0)
# snd_snd = PointMass(4, 2.0, 0.0)
# snd_thrd = PointMass(5, 4.0, 0.0)
#
# pointMassArr = [fst, fst_fst, fst_snd, snd_fst, snd_snd, snd_thrd]



# Normal distribution with following parameters, divided by 4 to be at 1:
# (1 / (0.0997356sqrt(2π)) ℯ^(-(1 / 2) ((x - sqrt(2)) / 0.0997356)²)) / 4
mu = math.sqrt(2)
omega = 0.0997356

score_scalar = 12
# failure_probability = 0.2
global failure_probability
global failure_scale


# pid = PID(1, 0.1, 0.05, setpoint=mu)
spring_constant = 78.872689 # kg/m
# damping_constant = 1.04171476 # kg s / m
damping_constant = 0
omega_d = math.sqrt(spring_constant - (damping_constant/2)**2) # damped natural frequency


def init_dots():
    fst = PointMass(0, 2.0, 2.0)
    fst_fst = PointMass(1, 1.0, 1.0)
    fst_snd = PointMass(2, 3.0, 1.0)
    snd_fst = PointMass(3, 0.0, 0.0)
    snd_snd = PointMass(4, 2.0, 0.0)
    snd_thrd = PointMass(5, 4.0, 0.0)

    return [fst, fst_fst, fst_snd, snd_fst, snd_snd, snd_thrd]


def score(x):
    try:
        res = max(0.0, score_scalar * abs(0.9999997002109396 - (1 / (omega * math.sqrt(2*math.pi))) * math.pow(math.e, 0.5 * ((x - mu) / omega)**2) / 4))
        return res
    except Exception:
        return max(0.0, score_scalar * abs(0.9999997002109396 - (1 / (omega * math.sqrt(2*math.pi))) * float('inf')))


def calc_score(pointMassArr):
    distances = [0] * nb_agents
    scores = [0] * nb_agents
    for i in range(nb_agents):
        distances[i] = calc_distances(i, pointMassArr)
        scores[i] = score(distances[i])
    return 1 - np.average(scores)


def calc_distances(ix, pointMassArr):
    if ix == 1 or ix == 2:
        dist = math.sqrt(
            (pointMassArr[ix].x - pointMassArr[0].x) ** 2 + (pointMassArr[ix].y - pointMassArr[0].y) ** 2)
    elif ix == 3 or ix == 4:
        dist = math.sqrt(
            (pointMassArr[ix].x - pointMassArr[1].x) ** 2 + (pointMassArr[ix].y - pointMassArr[1].y) ** 2)
    else:
        dist = math.sqrt(
            (pointMassArr[ix].x - pointMassArr[2].x) ** 2 + (pointMassArr[ix].y - pointMassArr[2].y) ** 2)
    return dist


def calc_vel(ix, pointMassArr):
    if ix != 0:
        # 1,2 -> 0
        # 3,4 -> 1
        # 5 -> 2
        dist = calc_distances(ix, pointMassArr)
        # print("is sqrt2 = " + str(dist == mu) + "pid = " + str(pid(dist)))
        # print(pid.components)
        # Failure within distance sensing for agent 4
        if ix == 4 and np.random.rand() < failure_probability:
            dist += failure_scale

        if dist > math.sqrt(2):
            return 0.55
        elif dist < math.sqrt(2):
            return 0.45
        else:
            return 0.5
    else:
        return 0.5


def calc_vel_spring(ix, pointMassArr):
    if ix != 0:
        # 1,2 -> 0
        # 3,4 -> 1
        # 5 -> 2
        dist = calc_distances(ix, pointMassArr)
        # Failure within distance sensing for agent 4
        if ix == 4 and np.random.rand() < failure_probability:
            dist += failure_scale
        # if ix == 4:
            # print(dist/math.sqrt(2) * 0.5)
        return dist/mu * 0.5
    else:
        return 0.5


def calc_pos_damped_spring(ix, pointMassArr, tick):
    if ix != 0:
        # 1,2 -> 0
        # 3,4 -> 1
        # 5 -> 2
        dist = calc_distances(ix, pointMassArr)
        # Failure within distance sensing for agent 4
        if ix == 4 and np.random.rand() < failure_probability:
            dist += failure_scale

        if dist != math.sqrt(2):
            x = (1/(2*omega_d))*(math.e**(-(damping_constant/2)*tick))*math.sin(omega_d*tick)
            return pointMassArr[ix].y + x + (dist/mu) * 0.5
            # return x
        else:
            return pointMassArr[ix].y + 0.5
    else:
        return pointMassArr[ix].y + 0.5


def update_damped_spring_system(pointMassArr, tick):
    curr_score = calc_score(pointMassArr)
    positions = [0] * nb_agents
    for i in range(nb_agents):
        positions[i] = calc_pos_damped_spring(i, pointMassArr,tick)
    for i in range(nb_agents):
        pointMassArr[i].y = positions[i]
        # plt.plot(pointMassArr[i].x, pointMassArr[i].y, 'ro')
    # plt.show()
    return max(0, curr_score)


def update(pointMassArr):
    curr_score = calc_score(pointMassArr)
    # print(curr_score)
    velocities = [0] * nb_agents
    for i in range(nb_agents):
        velocities[i] = calc_vel(i, pointMassArr)
        # velocities[i] = calc_vel_dampened(i, pointMassArr)
    for i in range(nb_agents):
        pointMassArr[i].y += velocities[i]
        # plt.plot(pointMassArr[i].x, pointMassArr[i].y, 'ro')
    # plt.show()
    return max(0, curr_score)


if __name__ == "__main__":
    # for j in range(nb_setpoints):
    #     failure_probability = 0.05 * j
    #     for k in range(nb_setpoints):
    #         failure_scale = 0.05 * k
    #         scores_out_run = [0] * 1000
    #         for i in range(1000):
    #             point_mass_arr = init_dots()
    #             curr_tick = 0
    #             scores = [0] * nb_ticks
    #             while curr_tick != nb_ticks:
    #                 curr_score = update(point_mass_arr)
    #                 scores[curr_tick] = curr_score
    #                 curr_tick += 1
    #             scores_out_run[i] = np.average(scores)
    #         print(str(j)+","+str(k)+","+str(np.average(scores_out_run)))




    # failure_scale = 0.5
    # failure_probability = 0.5
    # scores_out_run = [0] * 1
    # for i in range(1):
    #     point_mass_arr = init_dots()
    #     curr_tick = 0
    #     scores = [0] * nb_ticks
    #     while curr_tick != nb_ticks:
    #         curr_score = update(point_mass_arr)
    #         scores[curr_tick] = curr_score
    #         curr_tick += 1
    #     # print(np.average(scores))
    #     scores_out_run[i] = np.average(scores)
    # print(np.average(scores_out_run))


    ### main with damped spring system








    # failure_scale = 0.5
    # failure_probability = 0.5
    # scores_out_run = [0] * 1
    # for i in range(1):
    #     point_mass_arr = init_dots()
    #     curr_tick = 0
    #     scores = [0] * nb_ticks
    #     while curr_tick != nb_ticks:
    #         curr_score = update_damped_spring_system(point_mass_arr, curr_tick)
    #         scores[curr_tick] = curr_score
    #         curr_tick += 1
    #     # print(np.average(scores))
    #     scores_out_run[i] = np.average(scores)
    # print(np.average(scores_out_run))

    for j in range(nb_setpoints):
        failure_probability = 0.05 * j
        for k in range(nb_setpoints):
            failure_scale = 0.05 * k
            scores_out_run = [0] * 100
            for i in range(100):
                point_mass_arr = init_dots()
                curr_tick = 0
                scores = [0] * nb_ticks
                while curr_tick != nb_ticks:
                    curr_score = update_damped_spring_system(point_mass_arr, curr_tick)
                    scores[curr_tick] = curr_score
                    curr_tick += 1
                scores_out_run[i] = np.average(scores)
            print(str(j)+","+str(k)+","+str(np.average(scores_out_run)))
