# Microcosme Internship
This repertory contains two python pipelines and two zip files containing some .laz files.

## First Pipeline

This file is not operational since it needs some of the wrapper method of Microcosme company which is private property I can not had to the present public repertory.
It made availalbe for reading purpose only.

##Second Pipeline

This file can be executed as long as the following package are downloaded before hand :

\begin{itemize}
\item laspy
\item laszip
\item scipy
\item numpy
\item pandas
\item open3d
\end{itemize}

### Set-up

pip laspy[laszip]
pip scipy
pip scipy
pp numpy
pip pandas
pip open3d

### Execution

The second pipeline possess some lines to execute the .laz files contained in this repertory. If you execute the program using

python3 Pipeline2.py

It will crash if you have not beforehand changed : the sys path at the beginning of the file and if you did not extract the .laz files of set1 and set2 in the same repertory than Pipeline2

The best way to play with Pipeline2.py is to open it in a modern Editor and execute pipeline2(<las_file_path>,<nbr_points>) after compiling the rest of the file.

where
\begin{itemize}
  \item las_file_path : the path to the .laz/.las file to process
  \item nbr_points : the number of points of the .laz points to process (ATTENTION : if it is to big the code will crash 500 000 points will work with every file of set1 and set2 but many laz files can handle more points)
  \end{itemize}

The different hyperparameters of the pipeline are available at the beginning of the file they are originally set to the one chosen at the end of the report :

\begin{equation*}
    \begin{aligned}
    \text{Height factor :}&\\
    N_{neigh} &=15\\
    \sigma_{thresh}&=0.5\\
    \text{DBSCAN :}&\\
    \text{Radius}&= 1.1\\
    N_{neigh}&=5
    \end{aligned}
\end{equation*}
