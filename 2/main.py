import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
import warnings

"""
    Program klimatyzacja działa w taki sposób, że sprawdza:
     - temperaturę aktualną w pokoju
     - wilgotność powietrza w pokoju
     - prędkośc wiatru w pokoju
     i na podstawie tych warunków w danym momencie dostosowywuje wartość temperatury
      która powinna byc w pomieszczeniu

    Jak przygotwac srodowisko do uruchomienia programu:
     - Zainstalowac numpy - pip install numpy
     - Zainstalowac skfuzzy - pip install scikit-fuzzy
     - Zainstalowac warnings - pip install warnings
     - Zainstalowac matplotlib - pip install matplotlib

    Autorzy:
     - Jakub Gwiazda(s20497)
     - Wanda Bojanowska(s18425)
"""

from matplotlib import MatplotlibDeprecationWarning

"""
w trakcie wywoływania kodu pojawiaja się błedy do wykresów
MatplotlibDeprecationWarning: Support for FigureCanvases without 
a required_interactive_framework attribute was deprecated 
in Matplotlib 3.6 and will be removed two minor releases later.
"""
warnings.simplefilter(action='ignore', category=MatplotlibDeprecationWarning)


def setting_ranges():
    """
    ustawianie zakresów wejść:
     - temperatura pokoju
     - wilgotność
     - prędkość wiatru
    """
    global x_room_temp
    global x_humidity
    global x_speed

    global x_AC_temp

    x_room_temp = np.arange(14, 31, 1)
    x_humidity = np.arange(0, 101, 1)
    x_speed = np.arange(0, 1.1, 0.1)

    """
    ustawianie zakresu wyjściowej temperatury
    """
    x_AC_temp = np.arange(14, 31, 1)

def setting_fuzzy_ranges():
    """
    ustawianie wartości low,mid,high dla wartości:
     - temperatura pokoju
     - wilgotność
     - prędkość wiatru
     - temperatura wyjściowa
    """

    global room_temp_lo
    global room_temp_md
    global room_temp_hi

    global humidity_lo
    global humidity_md
    global humidity_hi

    global speed_lo
    global speed_md
    global speed_hi

    global AC_temp_lo
    global AC_temp_md
    global AC_temp_hi

    room_temp_lo = fuzz.trimf(x_room_temp, [0, 14, 22])
    room_temp_md = fuzz.trimf(x_room_temp, [14, 22, 30])
    room_temp_hi = fuzz.trimf(x_room_temp, [22, 30, 30])

    humidity_lo = fuzz.trimf(x_humidity, [0, 0, 50])
    humidity_md = fuzz.trimf(x_humidity, [0, 50, 100])
    humidity_hi = fuzz.trimf(x_humidity, [50, 100, 100])

    speed_lo = fuzz.trimf(x_speed, [0, 0, 0.5])
    speed_md = fuzz.trimf(x_speed, [0, 0.5, 1])
    speed_hi = fuzz.trimf(x_speed, [0.5, 1, 1])

    AC_temp_lo = fuzz.trimf(x_AC_temp, [0, 14, 22])
    AC_temp_md = fuzz.trimf(x_AC_temp, [14, 22, 30])
    AC_temp_hi = fuzz.trimf(x_AC_temp, [22, 30, 30])



def creating_plot_fuzzy_ranges():
    """
    ustawianie wartości wejściowych
    zakresy:
     - room_temp - 14-30
     - humidity - 0-100
     - air_speed - 0-1

     ustawianie zakresów dla fuzzy logic
    """

    global temp_level_lo
    global temp_level_md
    global temp_level_hi

    global humidity_level_lo
    global humidity_level_md
    global humidity_level_hi

    global speed_level_lo
    global speed_level_md
    global speed_level_hi


    temp_level_lo = fuzz.interp_membership(x_room_temp, room_temp_lo, room_temp_input)
    temp_level_md = fuzz.interp_membership(x_room_temp, room_temp_md, room_temp_input)
    temp_level_hi = fuzz.interp_membership(x_room_temp, room_temp_hi, room_temp_input)

    humidity_level_lo = fuzz.interp_membership(x_humidity, humidity_lo, humidity_input)
    humidity_level_md = fuzz.interp_membership(x_humidity, humidity_md, humidity_input)
    humidity_level_hi = fuzz.interp_membership(x_humidity, humidity_hi, humidity_input)

    speed_level_lo = fuzz.interp_membership(x_speed, speed_lo, air_speed_input)
    speed_level_md = fuzz.interp_membership(x_speed, speed_md, air_speed_input)
    speed_level_hi = fuzz.interp_membership(x_speed, speed_hi, air_speed_input)


def drawing_initial_plots_and_output_fuzzy_plot():
    """
    wyrysowywanie wykresów
    """
    # Visualize these universes and membership functions
    fig, (ax0, ax1, ax2) = plt.subplots(nrows=3, figsize=(8, 9))

    ax0.plot(x_room_temp, room_temp_lo, 'b', linewidth=1.5, label='Low')
    ax0.plot(x_room_temp, room_temp_md, 'g', linewidth=1.5, label='Medium')
    ax0.plot(x_room_temp, room_temp_hi, 'r', linewidth=1.5, label='High')
    ax0.plot([room_temp_input, room_temp_input], [0, 1], 'k', linewidth=3.5, alpha=0.9)
    ax0.set_title('Initial room temperature')
    ax0.legend()


    ax1.plot(x_humidity, humidity_lo, 'b', linewidth=1.5, label='Low')
    ax1.plot(x_humidity, humidity_md, 'g', linewidth=1.5, label='Medium')
    ax1.plot(x_humidity, humidity_hi, 'r', linewidth=1.5, label='High')
    ax1.plot([humidity_input, humidity_input], [0, 1], 'k', linewidth=3.5, alpha=0.9)
    ax1.set_title('Humidity in room')
    ax1.legend()

    ax2.plot(x_speed, speed_lo, 'b', linewidth=1.5, label='Low')
    ax2.plot(x_speed, speed_md, 'g', linewidth=1.5, label='Medium')
    ax2.plot(x_speed, speed_hi, 'r', linewidth=1.5, label='High')
    ax2.plot([air_speed_input, air_speed_input], [0, 1], 'k', linewidth=3.5, alpha=0.9)
    ax2.set_title('Air speed in room')
    ax2.legend()

    # Turn off top/right axes
    for ax in(ax0, ax1, ax2):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    plt.tight_layout()

    plt.show()



def setting_rules_for_output():
    """
    ustawianie zasad

    1. If room temp is low and air speed is low then AC temp is high
    2. If room temp is medium and humidity is high then AC temp is med
    3. If room temp is high and air speed is high then AC temp is low
    """

    global temp_activation_lo
    global temp_activation_md
    global temp_activation_hi

    active_rule_hi = np.fmax(temp_level_lo, speed_level_lo)

    temp_activation_hi = np.fmin(active_rule_hi, AC_temp_hi)

    active_rule_md = np.fmax(temp_level_md, humidity_level_hi)
    temp_activation_md = np.fmin(active_rule_md, AC_temp_md)

    active_rule_lo = np.fmax(temp_level_hi, speed_level_hi)
    temp_activation_lo = np.fmin(active_rule_lo, AC_temp_lo)


def drawing_output_membership_activity_plot():
    """
    wyrysowywanie wykresów
    """
    ACtemp0 = np.zeros_like(x_AC_temp)
    # Visualize this
    fig, membership_plot = plt.subplots(figsize=(8, 3))

    membership_plot.fill_between(x_AC_temp, ACtemp0, temp_activation_lo, facecolor='b', alpha=0.7)
    membership_plot.plot(x_AC_temp, AC_temp_lo, 'b', linewidth=0.5, linestyle='--', label='Low')
    membership_plot.fill_between(x_AC_temp, ACtemp0, temp_activation_md, facecolor='g', alpha=0.7)
    membership_plot.plot(x_AC_temp, AC_temp_md, 'g', linewidth=0.5, linestyle='--', label='Medium')
    membership_plot.fill_between(x_AC_temp, ACtemp0, temp_activation_hi, facecolor='r', alpha=0.7)
    membership_plot.plot(x_AC_temp, AC_temp_hi, 'r', linewidth=0.5, linestyle='--', label='High')
    membership_plot.set_title('Output membership activity')
    # Turn off top/right axes
    for ax in (membership_plot,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()

    plt.tight_layout()
    plt.show()


def drawing_membership_and_result_plot():
    """
    wyrysowywanie wykresów
    """
    ACtemp0 = np.zeros_like(x_AC_temp)
    # Aggregate all three output membership functions together
    aggregated = np.fmax(temp_activation_lo, np.fmax(temp_activation_md, temp_activation_hi))
    # Calculate defuzzified result
    ACtemp = fuzz.defuzz(x_AC_temp, aggregated, 'centroid')

    print('Temperatura ustawiona przez klimatyzacje: ' + str(ACtemp))

    temperature_activation = fuzz.interp_membership(x_AC_temp, aggregated, ACtemp)

    fig, membership_plot = plt.subplots(figsize=(8, 3))
    membership_plot.plot(x_AC_temp, AC_temp_lo, 'b', linewidth=1, linestyle='-', label='Low')
    membership_plot.plot(x_AC_temp, AC_temp_md, 'g', linewidth=1, linestyle='-', label='Medium')
    membership_plot.plot(x_AC_temp, AC_temp_hi, 'r', linewidth=1, linestyle='-', label='High')
    membership_plot.fill_between(x_AC_temp, ACtemp0, aggregated, facecolor='Orange', alpha=0.7)
    membership_plot.plot([ACtemp, ACtemp], [0, temperature_activation], 'k', linewidth=3.5, alpha=0.9)
    membership_plot.set_title('Aggregated membership and AC temp result (line)')
    membership_plot.legend()
    # Turn off top/right axes
    for ax in (membership_plot,):
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.get_xaxis().tick_bottom()
        ax.get_yaxis().tick_left()
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    """
    wejscia programu
    """
    room_temp_input = 29
    humidity_input = 30
    air_speed_input = 0.9

    """
    kod został podzielony na funkcje żeby był bardziej czytelny
    """

    """
    ustawianie zakresów wejść
    """
    setting_ranges()

    """
    ustawianie wartości low,mid,high dla wejść i wyjść
    """
    setting_fuzzy_ranges()

    """
    ustawianie zakresów dla fuzzy logic
    """
    creating_plot_fuzzy_ranges()

    """
    wyrysowanie diagramów wejściowych wraz z zaznaczeniem wartości wejściowych
    """
    drawing_initial_plots_and_output_fuzzy_plot()

    """
    ustawienie zasad dla klimatyzatora
    """
    setting_rules_for_output()

    """
    wyrysowanie diagramu Output membership activity
    """
    drawing_output_membership_activity_plot()

    """
    wyrysowanie diagramu wyjściowej temperatury
    """
    drawing_membership_and_result_plot()


