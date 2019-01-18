import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib
import sys

tableau20 = [(31, 119, 180), (174, 199, 232), (255, 127, 14), (255, 187, 120),    
			(44, 160, 44), (152, 223, 138), (214, 39, 40), (255, 152, 150),    
			(148, 103, 189), (197, 176, 213), (140, 86, 75), (196, 156, 148),    
			(227, 119, 194), (247, 182, 210), (127, 127, 127), (199, 199, 199),    
			(188, 189, 34), (219, 219, 141), (23, 190, 207), (158, 218, 229)]    

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
#matplotlib.rcParams['text.usetex'] = True
matplotlib.rcParams['axes.unicode_minus']=False
matplotlib.rcParams.update({'figure.autolayout':True})
matplotlib.rcParams.update({'font.size': 18})
matplotlib.rc('pdf', fonttype=42)

def plot_line(fname, xname, yname, xdata, ydata, xlim=None, ylim=None, xticks=None, yticks=None, xlabels=None, ylabels=None, fsize=None, marks=None, legends=None, marker=None, ymin=None, loc=None, fontsize=18):
	# write_file(fname, xname, yname, xdata, ydata, legends)
	plt.clf()
	fs = (8,3)
	if fsize is not None:
		fs = fsize
	fig = plt.figure(figsize=fs)
	pp = PdfPages(fname)
	markers=['o','v','^','<','>','.',',','1','2','3','4','8','s','p','P','*','h','H','+','x','X','D','d','|','_']
	linestyle=['-','--',':','-.']
	if marks is not None:
		mi = 0
		for m in marks:
			plt.scatter(m[0], m[1], marker=markers[mi], color='red')
			# plt.scatter(m[2], m[3], marker=markers[mi], color='red')
			mi = mi + 1
			mi = mi % len(markers)
	if legends is not None:
		for i in range(0, len(legends)):
			# print xdata[i]
			# print ydata[i]
			# print legends[i]
			if marker is not None:
				plt.plot(xdata[i], ydata[i], label=legends[i], marker=marker, linewidth=4.0, linestyle=linestyle[i])
			else:
				plt.plot(xdata[i], ydata[i], label=legends[i], linewidth=4.0, linestyle=linestyle[i])
		if loc is not None:
			plt.legend(loc=loc, prop={'size': fontsize})
		else:
			plt.legend(loc="lower right", prop={'size': fontsize})
	else:
		if marker is not None:
			plt.plot(xdata, ydata, marker=marker)
		else:
			plt.plot(xdata, ydata)
	plt.xlabel(xname, fontsize=fontsize)
	plt.ylabel(yname, fontsize=fontsize)
	if xlim is not None:
		plt.ylim(xlim)
	if ylim is not None:
		plt.ylim(ylim)
	if xticks is not None:
		if xlabels is not None:
			plt.xticks(xticks, xlabels, fontsize=fontsize)
		else:
			plt.xticks(xticks, fontsize=fontsize)
	else:
		plt.xticks(fontsize=fontsize)
	if yticks is not None:
		if ylabels is not None:
			plt.yticks(yticks, ylabels, fontsize=fontsize)
		else:
			plt.yticks(yticks, fontsize=fontsize)
	else:
		plt.xticks(fontsize=fontsize)
	if ymin is not None:
		plt.ylim(ymin=ymin)
	pp.savefig(bbox_inches='tight')
	pp.close()

def get_results(fname=None):
    x = []
    y = []
    instance_num = 1
    switched_to_exploit = 0
    instance_num_limit = 10000
    if fname == None:
        fname = "output"
    file = open(fname, "r")
    for line in file:
        line = line[:-1]
        if line.startswith("TIME"):
            line_split = line.split(" ")
            if len(line_split) == 4:
                x.append(instance_num)
                y.append(float(line_split[-1]))
                instance_num = instance_num + 1
        if line.startswith("Switching"):
            switched_to_exploit = instance_num
        if instance_num >= instance_num_limit:
            break
    plot_line(fname+".pdf", 'Instance Number', 'Deviation', x, y, marks=[(switched_to_exploit, 0)])
    file.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        fname = sys.argv[1]
        get_results(fname=fname)
    else:
        get_results()