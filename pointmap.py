from multiprocessing import Process, Queue
import numpy as np
import vtk
from vtkmodules.util import numpy_support

#Global visualization usng vtk
class Map():
    def __init__(self):
        self.frames = []  # camera frames [synonymous to camera pose]
        self.points = [] # 3D points of map
        self.state = None # variable to store current state of the map and the camera pose
        self.q = None # A queue for inter-process communication

    def create_viewer(self):
        #parallel ececution to run viewer_thread in parallel with main program

        self.q = Queue() 
        #args here is an iteratable, where we pass the queue object
        p = Process(target = self.viewer_thread, args=(self.q,))

        #daemon=True -> exit when the main program stops
        p.daemon = True

        # start the proces
        p.start()

    def viewer_thread(self, q):
        #'viewer_thread' takes the q as input
        #initializes the viz window
        self.viewer_init(1280, 720)
        while True:
            self.viewer_refresh(q)

    def viewer_init(self, w, h):
        
        #create a renderer, render window and interactor
        self.renderer = vtk.vtkRenderer()
        self.renderer.SetBackground(1.0, 1.0, 1.0) # White background

        self.render_window = vtk.vtkRenderWindow()
        self.render_window.AddRenderer(self.renderer)
        self.render_window.SetSize(w, h)

        self.interactor = vtk.vtkRenderWindowInteractor()
        self.interactor.SetRenderWindow(self.render_window)

        #set up camera
        self.camera = self.renderer.GetActiveCamera()
        self.camera.SetPosition(0, -10, -8)
        self.camera.SetFocalPoint(0, 0, 0)
        self.camera.SetViewUp(0, -1, 0)

        #Set up interactor style for 3D Navigation
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.interactor.SetInteractorStyle(style)

        #initialize the camera path actor
        self.camera_path_actor = vtk.vtkActor()
        self.renderer.AddActor(self.camera_path_actor)

        #initialize the points actor
        self.points_actor = vtk.vtkActor()
        self.renderer.AddActor(self.points_actor)

        #start the interactor
        self.interactor.Initialize()
        self.render_window.Render()

        #set up a timer callback for continuous updates
        self.interactor.CreateRepeatingTimer(10)

    def viewer_refresh(self, q: Queue):
        #check if there is new data in the queue
        if self.state is None or not q.empty():
            self.state = q.get

            # Update camera trajectory
            if len(self.state[0]) > 0:
                self._update_camera_path(self.state[0])
            
            # Update point cloud
            if len(self.state[1]) > 0:
                self._update_points(self.state[1])
        
        # Render the scene
        self.render_window.Render()
        self.interactor.ProcessEvents()

    def _update_camera_path(self, poses):

        # Create a polydata for the camera path
        points = vtk.vtkPoints()
        lines = vtk.vtkCellArray()
        
        # Add points for each camera pose
        for i, pose in enumerate(poses):
            # Extract translation part of the pose
            position = pose[:3, 3]
            points.InsertNextPoint(position[0], position[1], position[2])
            
            # Create a line segment between consecutive points
            if i > 0:
                line = vtk.vtkLine()
                line.GetPointIds().SetId(0, i-1)
                line.GetPointIds().SetId(1, i)
                lines.InsertNextCell(line)
                
            #add camera frustum visualization here
            self._add_camera_frustum(pose, points, lines, i)
        
        # Create a polydata object
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        polydata.SetLines(lines)
        
        # Create a mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputData(polydata)
        
        # Set the actor properties
        self.camera_path_actor.SetMapper(mapper)
        self.camera_path_actor.GetProperty().SetColor(0.0, 1.0, 0.0)  # Green
        self.camera_path_actor.GetProperty().SetLineWidth(2)

    def _add_camera_frustum(self, pose, points, lines, base_index):
        # This is a simplified camera frustum 
        scale = 0.5  # Scale of the frustum
        
        # Camera center
        center = pose[:3, 3]
        
        # Camera axes
        x_axis = pose[:3, 0] * scale
        y_axis = pose[:3, 1] * scale
        z_axis = pose[:3, 2] * scale
        
        # Define the four corners of the frustum
        corners = [
            center + z_axis + x_axis + y_axis,
            center + z_axis - x_axis + y_axis,
            center + z_axis - x_axis - y_axis,
            center + z_axis + x_axis - y_axis
        ]
        
        # Add the corners as points
        corner_indices = []
        for corner in corners:
            idx = points.InsertNextPoint(corner[0], corner[1], corner[2])
            corner_indices.append(idx)
        
        # Add lines from center to corners
        center_idx = base_index
        for corner_idx in corner_indices:
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, center_idx)
            line.GetPointIds().SetId(1, corner_idx)
            lines.InsertNextCell(line)
        
        # Add lines between corners to form a frustum
        for i in range(4):
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, corner_indices[i])
            line.GetPointIds().SetId(1, corner_indices[(i+1) % 4])
            lines.InsertNextCell(line)

    def _update_points(self, pts):
        # Create a vtkPoints object and populate it with the 3D points
        points = vtk.vtkPoints()
        for pt in pts:
            points.InsertNextPoint(pt[0], pt[1], pt[2])
        
        # Create a polydata object
        polydata = vtk.vtkPolyData()
        polydata.SetPoints(points)
        
        # Use vtkVertexGlyphFilter to render points
        vertex_filter = vtk.vtkVertexGlyphFilter()
        vertex_filter.SetInputData(polydata)
        vertex_filter.Update()
        
        # Create mapper and actor
        mapper = vtk.vtkPolyDataMapper()
        mapper.SetInputConnection(vertex_filter.GetOutputPort())
        
        # Set the actor properties
        self.points_actor.SetMapper(mapper)
        self.points_actor.GetProperty().SetColor(1.0, 0.0, 0.0)  # Red
        self.points_actor.GetProperty().SetPointSize(3)

    
    def display(self):
        if self.q is None:
            return
        
        poses, pts = [], []
        for f in self.frames:
            # updating pose
            poses.append(f.pose)
        
        for p in self.points:
            # updating map points
            pts.append(p.pt)
        
        # updating queue
        self.q.put((np.array(poses), np.array(pts)))


class Point():
    #A point is a 3d point in the world
    # Each point is observed in multiple frames
    def __init__(self, mapp:Map, loc):
        self.frames = []
        self.pt = loc
        self.idxs = []

        #assigns a uniqye id to the point based on num of points in the map
        self.id = len(mapp.points)
        #adds the points instance to the map's list of points
        mapp.points.append(self)


    def add_observation(self, frame, idx):
        self.frames.append(frame)
        self.idxs.append(idx)