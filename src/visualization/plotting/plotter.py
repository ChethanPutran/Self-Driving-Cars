import pandas as pd
from bokeh.plotting import figure
from bokeh.io import output_file, show
from bokeh.layouts import gridplot
from bokeh.models import HoverTool,ColumnDataSource
from datetime import datetime

class Plot:
    
    def __init__(self,fileName):
        self.fileName = fileName
        
        
    #Getting plotting Data
    def getDataFromFile(self):
        
        df = pd.DataFrame(columns=["Entered Time","Left Time"])

        with open(self.fileName,'r') as destination:       
            lines = destination.readlines()
            
        for i in range(1,len(lines)):
            line = lines[i].split(',')
            start = line[1]
            start = datetime.strptime(start, '%Y-%m-%d %H:%M:%S.%f')
            end = line[2].rstrip('\n')
            end = datetime.strptime(end, '%Y-%m-%d %H:%M:%S.%f')
            df = df.append({"Entered Time": start,"Left Time": end},ignore_index=True)
            
        return df

    def makePlot(self):
        df = self.getDataFromFile()
        
        df['Start_string'] = df['Entered Time'].dt.strftime("%Y-%m-%d %H:%M:%S")
        df['End_string'] = df['Left Time'].dt.strftime("%Y-%m-%d %H:%M:%S")
        columnDS = ColumnDataSource(df)

        #Creatingb a figure
        plot = figure(x_axis_type ="datetime",height=100,width=500,title="Motion Graph")
        plot.yaxis[0].ticker.desired_num_ticks = 1
        plot.yaxis.minor_tick_line_color = None

        hover = HoverTool(tooltips=[('Entered Time',"@Start_string"),('Left Time',"@End_string")])
        plot.add_tools(hover)
        plot.quad(left="Entered Time",right="Left Time",bottom=0,top=1,color='green',source=columnDS)
        plot = gridplot([[plot]], sizing_mode='scale_width')
        output_file("Graph.html")

        #Adding a grid layout

        show(plot)

plot = Plot("motion.csv")
plot.makePlot()
