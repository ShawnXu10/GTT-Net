import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

def update_lines(num, dataLines, lines, limbs, limbSeq):
    for line, data in zip(lines, dataLines):
        # NOTE: there is no .set_data() for 3 dim data...
        line.set_data(data[0:2, :num])
        line.set_3d_properties(data[2, :num])
    for limb, i in zip(limbs, range(limbSeq.shape[0])):
        limb.set_data(np.array([[dataLines[limbSeq[i,0]][0,num], dataLines[limbSeq[i,1]][0,num]],
                                   [dataLines[limbSeq[i,0]][1,num], dataLines[limbSeq[i,1]][1,num]]]))
        limb.set_3d_properties(np.array([dataLines[limbSeq[i,0]][2,num], dataLines[limbSeq[i,1]][2,num]]))
def Animate3DSeleton(point3D, bounds):
    
    fig = plt.figure()
    ax = p3.Axes3D(fig)
    if len(point3D.shape)==2:
        nframes = point3D.shape[1]
        njoints = int(point3D.shape[0]/3)
        data = [point3D[i*3:i*3+3,:] for i in range(njoints)]
    elif len(point3D.shape)==3:
        nframes = point3D.shape[2]
        njoints = point3D.shape[0]
        data = [point3D[i,:,:].transpose(0,1) for i in range(njoints)]

    if njoints == 31:
        limbSeq = np.array([[0, 1],[1, 2],[2, 3],[3, 4],[4, 5],
                            [0, 6],[6, 7],[7, 8],[8, 9],[9, 10],
                            [0, 11],[11, 12],[12, 13],[13, 14],[14, 15],[15, 16],   
                            [13, 17],[17, 18],[18, 19],[19, 20],[20, 21],[21, 22],
                            [20, 23],
                            [13, 24],[24, 25],[25, 26],[26, 27],[27, 28],[28, 29],
                            [27, 30]])
    else:
        limbSeq = np.array([[0, 0]])

    lines = [ax.plot(dat[0, 0:1], dat[1, 0:1], dat[2, 0:1], color = 'blue')[0] for dat in data]
    limbs = [ax.plot(np.array([data[limbSeq[i,0]][0,0], data[limbSeq[i,1]][0,0]]), np.array([data[limbSeq[i,0]][1,0], data[limbSeq[i,1]][1,0]]), np.array([data[limbSeq[i,0]][2,0], data[limbSeq[i,1]][2,0]]), color='red')[0] for i in range(limbSeq.shape[0])]

    # Setting the axes properties
    ax.set_xlabel('X')
    ax.set_xlim3d([bounds[0], bounds[1]])
    ax.set_ylabel('Y')
    ax.set_ylim3d([bounds[2], bounds[3]])
    ax.set_zlabel('Z')
    ax.set_zlim3d([bounds[4], bounds[5]])
    ax.set_title('3D Test')

    line_animation = animation.FuncAnimation(fig, update_lines, nframes, fargs=(data, lines,limbs, limbSeq),
                                   interval=50, blit=False)
    plt.show()