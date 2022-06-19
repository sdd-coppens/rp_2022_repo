import math

import numpy as np
from dataclasses import dataclass
from matplotlib import pyplot as plt
from matplotlib.animation import FuncAnimation
import imageio


# fig, ax = plt.subplots()
# ax = plt.axes(xlim=(-0.5, 4.5), ylim=(0, 5))
# tempA, = ax.plot([], [], 'ro')
# tempB, = ax.plot([], [], 'ro')
# tempC, = ax.plot([], [], 'ro')
# tempD, = ax.plot([], [], 'ro')
# tempE, = ax.plot([], [], 'ro')
# tempF, = ax.plot([], [], 'ro')


optimising_test = False
graphing = False
delay_test = True


@dataclass
class PointMass:
    id: str
    x: float
    y: float
    vel_x: float
    vel_y: float
    a_x: float
    a_y: float


mu = math.sqrt(2)
omega = 0.0997356

score_scalar = 12


nb_ticks = 2000
tick_size = 0.001
global curr_tick
nb_setpoints = 21
global failure_probability
global failure_scale
nb_agents = 6

spr_rest = (1, 1)
global spr_const
global damp_const_derived
# spr_const = 1
# damp_const_derived = 99
global delay_arr
global delay_amount
default_pos = [('A', 2.0, 2.0), ('B', 1.0, 1.0), ('C', 3.0, 1.0), ('D', 0.0, 0.0), ('E', 2.0, 0.0), ('F', 4.0, 0.0)]


def F_ij(xi1, xi2, xj1, xj2):
    (dist1, dist2) = ((xi1 - xj1), (xi2 - xj2))
    (term11, term12) = (dist1/abs(dist1), dist2/abs(dist2))
    (term21, term22) = (spr_const * (abs(dist1) - 1), spr_const * (abs(dist2) - 1))
    return ((term11 - term21), (term12 - term22))


def calc_accel(id, point_mass_arr, neighbours, curr_tick, delay_arr):
    (sumx, sumy) = (0, 0)
    xi1 = point_mass_arr[ord(id)-ord('A')].x
    xi2 = point_mass_arr[ord(id)-ord('A')].y
    velx = point_mass_arr[ord(id)-ord('A')].vel_x
    vely = point_mass_arr[ord(id)-ord('A')].vel_y
    for neighbour_id in neighbours[ord(id)-ord('A')]:
        xj1 = point_mass_arr[ord(neighbour_id)-ord('A')].x
        xj2 = point_mass_arr[ord(neighbour_id)-ord('A')].y

        if 4 == (ord(id)-ord('A')) and np.random.rand() < failure_probability:
            xj2 += failure_scale

        # if delay_test and 4 == (ord(id)-ord('A')) and curr_tick >= delay_amount:
        if delay_test and curr_tick >= delay_amount:
            delay_arr[(ord(id)-ord('A'))][ord(neighbour_id)-ord('A')][curr_tick] = (xj1, xj2)

            xj1 = delay_arr[(ord(id)-ord('A'))][ord(neighbour_id)-ord('A')][curr_tick - delay_amount][0]
            xj2 = delay_arr[(ord(id)-ord('A'))][ord(neighbour_id)-ord('A')][curr_tick - delay_amount][1]
        elif delay_test:
            xj1 = default_pos[(ord(neighbour_id)-ord('A'))][1]
            xj2 = default_pos[(ord(neighbour_id) - ord('A'))][2]

        (sumxloop, sumyloop) = F_ij(xi1, xi2, xj1, xj2)
        # working ish
        sumx += abs(sumxloop)
        sumy += abs(sumyloop)
        # sumx += sumxloop
        # sumy += sumyloop

    sumx /= len(neighbours[ord(id)-ord('A')])
    sumy /= len(neighbours[ord(id)-ord('A')])
    (term2x, term2y) = (damp_const_derived * velx, damp_const_derived * vely)
    return ((sumx - term2x), (sumy - term2y))


def init_dots():
    fst = PointMass('A', 2.0, 2.0, 0, 0.5, 0, 0)       # A
    fst_fst = PointMass('B', 1.0, 1.0, 0, 0.5, 0, 0)   # B
    fst_snd = PointMass('C', 3.0, 1.0, 0, 0.5, 0, 0)   # C
    snd_fst = PointMass('D', 0.0, 0.0, 0, 0.5, 0, 0)   # D
    snd_snd = PointMass('E', 2.0, 0.0, 0, 0.5, 0, 0)   # E
    snd_thrd = PointMass('F', 4.0, 0.0, 0, 0.5, 0, 0)  # F

    neighbours = [['B', 'C'], ['A', 'D', 'E'], ['A', 'E', 'F'], ['B'], ['B', 'C'], ['C']]
    return ([fst, fst_fst, fst_snd, snd_fst, snd_snd, snd_thrd], neighbours)


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


def score(x):
    try:
        res = max(0.0, score_scalar * abs(0.9999997002109396 - (1 / (omega * math.sqrt(2*math.pi))) * math.pow(math.e, 0.5 * ((x - mu) / omega)**2) / 4))
        return res
    except Exception:
        return max(0.0, score_scalar * abs(0.9999997002109396 - (1 / (omega * math.sqrt(2*math.pi))) * float('inf')))


def calc_score(pointMassArr):
    distances = [0] * nb_agents
    scores = [0] * nb_agents
    for ij in range(nb_agents):
        distances[ij] = calc_distances(ij, pointMassArr)
        scores[ij] = score(distances[ij])
    return 1 - np.average(scores)


if __name__ == "__main__":
    # Best result
    # (1, 99, 0.9999445984695993)
    if optimising_test and not graphing and not delay_test:
        best_comb = (0, 0, 0)
        failure_scale = 1
        failure_probability = 1
        # j is spring const
        for j in range(1, 100):
            spr_const = j
            # k is dampening constant
            for k in range(1, 100):
                damp_const_derived = k

                scores_out_run = [0] * 1
                for run_n in range(1):
                    (point_mass_arr, neighbours) = init_dots()
                    curr_tick = 0
                    scores = [0] * nb_ticks
                    while curr_tick != nb_ticks:
                        accels = [(0, 0)] * nb_agents
                        vels = [(0, 0)] * nb_agents
                        for i in range(nb_agents):
                            accel = calc_accel(point_mass_arr[i].id, point_mass_arr, neighbours, 0, [])
                            vel = ((accel[0] * tick_size), (accel[1] * tick_size))  # Integrate accel over time to get vel.
                            accels[i] = accel
                            vels[i] = vel
                        for i in range(nb_agents):
                            point_mass_arr[i].vel_x += vels[i][0]
                            point_mass_arr[i].vel_y = 0.5 + vels[i][1]

                            point_mass_arr[i].x += tick_size * point_mass_arr[i].vel_x
                            point_mass_arr[i].y += tick_size * point_mass_arr[i].vel_y

                        curr_score = calc_score(point_mass_arr)
                        scores[curr_tick] = curr_score
                        curr_tick += 1
                    scores_out_run[run_n] = np.average(scores)
                #print(str(j)+","+str(k)+","+str(np.average(scores_out_run)))
                if np.average(scores_out_run) > best_comb[2]:
                    best_comb = (j, k, np.average(scores_out_run))
        print(best_comb)
    elif not graphing and not optimising_test and not delay_test:
        spr_const = 101
        damp_const_derived = 20
        for j in range(nb_setpoints):
            failure_probability = 0.05 * j
            for k in range(nb_setpoints):
                failure_scale = 0.05 * k

                # failure_scale = 1
                # failure_probability = 1

                scores_out_run = [0] * 100
                for run_n in range(100):
                    (point_mass_arr, neighbours) = init_dots()
                    curr_tick = 0
                    scores = [0] * nb_ticks
                    while curr_tick != nb_ticks:
                        accels = [(0, 0)] * nb_agents
                        vels = [(0, 0)] * nb_agents
                        for i in range(nb_agents):
                            accel = calc_accel(point_mass_arr[i].id, point_mass_arr, neighbours, 0, [])
                            vel = ((accel[0] * tick_size), (accel[1] * tick_size))  # Integrate accel over time to get vel.
                            accels[i] = accel
                            vels[i] = vel
                        for i in range(nb_agents):
                            point_mass_arr[i].vel_x += vels[i][0]
                            point_mass_arr[i].vel_y = 0.5 + vels[i][1]

                            point_mass_arr[i].x += tick_size * point_mass_arr[i].vel_x
                            point_mass_arr[i].y += tick_size * point_mass_arr[i].vel_y

                        curr_score = calc_score(point_mass_arr)
                        scores[curr_tick] = curr_score
                        curr_tick += 1
                    scores_out_run[run_n] = np.average(scores)
                print(str(failure_probability)+","+str(failure_scale)+","+str(np.average(scores_out_run)))
    elif graphing and not optimising_test and not delay_test:
        # graphing
        (point_mass_arr, neighbours) = init_dots()
        curr_tick = 0
        scores = [0] * nb_ticks
        failure_scale = 1
        failure_probability = 1
        spr_const = 100
        damp_const_derived = 20
        images = []
        filenames = []
        for i in range(nb_ticks):
            filenames.append('images_bad_params/'+str(i+1)+'.png')
        while curr_tick != nb_ticks:
            accels = [(0, 0)] * nb_agents
            vels = [(0, 0)] * nb_agents
            for i in range(nb_agents):
                accel = calc_accel(point_mass_arr[i].id, point_mass_arr, neighbours, 0, [])
                vel = ((accel[0] * tick_size), (accel[1] * tick_size))  # Integrate accel over time to get vel.
                accels[i] = accel
                vels[i] = vel
            for i in range(nb_agents):
                point_mass_arr[i].vel_x = vels[i][0]
                point_mass_arr[i].vel_y = 0.5 + vels[i][1]

                point_mass_arr[i].x += tick_size * point_mass_arr[i].vel_x
                point_mass_arr[i].y += tick_size * point_mass_arr[i].vel_y
            curr_score = calc_score(point_mass_arr)
            scores[curr_tick] = curr_score
            curr_tick += 1
        print(np.average(scores))

    else:
        for delay_count in range(450):
            (point_mass_arr, neighbours) = init_dots()
            curr_tick = 0
            scores = [0] * nb_ticks
            failure_scale = 0
            failure_probability = 0
            spr_const = 101
            damp_const_derived = 20
            # Distance stored as i -> j
            delay_arr = [[[(0, 0)] * nb_ticks for _ in range(nb_agents)] for _ in range(nb_agents)]
            # delay_arr = [(0, 0)] * nb_ticks
            delay_amount = delay_count

            while curr_tick != nb_ticks:
                accels = [(0, 0)] * nb_agents
                vels = [(0, 0)] * nb_agents
                for i in range(nb_agents):
                    accel = calc_accel(point_mass_arr[i].id, point_mass_arr, neighbours, curr_tick, delay_arr)
                    vel = ((accel[0] * tick_size), (accel[1] * tick_size))  # Integrate accel over time to get vel.
                    accels[i] = accel
                    vels[i] = vel
                for i in range(nb_agents):
                    point_mass_arr[i].vel_x = vels[i][0]
                    point_mass_arr[i].vel_y = 0.5 + vels[i][1]

                    point_mass_arr[i].x += tick_size * point_mass_arr[i].vel_x
                    point_mass_arr[i].y += tick_size * point_mass_arr[i].vel_y
                curr_score = calc_score(point_mass_arr)
                scores[curr_tick] = curr_score
                curr_tick += 1
            print(str(delay_count) + ", " + str(0 if np.average(scores) < 0 else np.average(scores)))
