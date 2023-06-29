import io
import matplotlib
import matplotlib.pyplot as plt
matplotlib.use('Agg')

def plot_2d(title, data):
    plt.figure()
    plt.plot(data[:, 0], data[:, 1])
    plt.title(title)
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    return buf