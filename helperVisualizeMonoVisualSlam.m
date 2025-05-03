function mapPointsPlot = helperVisualizeMonoVisualSlam(monoSlamOut)
%helperVisualizeMonoSlamResults to visualize the results of codegen output
%of mono slam example

%   Copyright 2023 The MathWorks, Inc.

for i=1:numel(monoSlamOut.estimatedPoses)
    posesArray(i, 1) = rigidtform3d(monoSlamOut.estimatedPoses{i});
end
for i=1:numel(monoSlamOut.optimizedPoses)
    optimizedPosesArray(i, 1) = rigidtform3d(monoSlamOut.optimizedPoses{i});
end
trajectory = vertcat(posesArray.Translation);
optimizedTrajectory = vertcat(optimizedPosesArray.Translation);
xyzPoints = monoSlamOut.xyzPoints;
XLim = [-1.5 1.5];
YLim = [-1 0.5];
ZLim = [-0.5 2];
mapPointsPlot = pcplayer(XLim, YLim, ZLim, ...
    'VerticalAxis', 'y', 'VerticalAxisDir', 'down');

mapPlotAxes  = mapPointsPlot.Axes;
mapPointsPlot.view(xyzPoints);
mapPlotAxes.Children.DisplayName = 'Map points';

hold(mapPlotAxes, 'on');
movegui(mapPlotAxes, [1000 200]);

% Plot camera trajectory
estimatedTrajectoryPlot = plot3(mapPlotAxes, trajectory(:,1), trajectory(:,2), ...
    trajectory(:,3), 'r', 'LineWidth', 2 , 'DisplayName', 'Estimated trajectory');
optimizedTrajectoryPlot = plot3(mapPlotAxes, optimizedTrajectory(:,1), optimizedTrajectory(:,2), ...
    optimizedTrajectory(:,3), 'm', 'LineWidth', 2 , 'DisplayName', 'Optimized trajectory');

gTruthData = load("orbslamGroundTruth.mat");
gTruthFull     = gTruthData.gTruth;
gTruth = gTruthFull(monoSlamOut.addedFramesIdx);
actualCams    = vertcat(gTruth.Translation);
scale = 2.6112;
mapPlotAxes.XLim = mapPlotAxes.XLim * scale;
mapPlotAxes.YLim = mapPlotAxes.YLim * scale;
mapPlotAxes.ZLim = mapPlotAxes.ZLim * scale;

% Map points
mapPlotAxes.Children(end).XData = mapPlotAxes.Children(end).XData * scale;
mapPlotAxes.Children(end).YData = mapPlotAxes.Children(end).YData * scale;
mapPlotAxes.Children(end).ZData = mapPlotAxes.Children(end).ZData * scale;

% Scale Estimated Camera trajectory
estimatedTrajectoryPlot.XData =  estimatedTrajectoryPlot.XData * scale;
estimatedTrajectoryPlot.YData =  estimatedTrajectoryPlot.YData * scale;
estimatedTrajectoryPlot.ZData =  estimatedTrajectoryPlot.ZData * scale;
optimizedTrajectoryPlot.XData =  optimizedTrajectoryPlot.XData * scale;
optimizedTrajectoryPlot.YData =  optimizedTrajectoryPlot.YData * scale;
optimizedTrajectoryPlot.ZData =  optimizedTrajectoryPlot.ZData * scale;
plot3(mapPlotAxes, actualCams(:,1), actualCams(:,2), actualCams(:,3), ...
    'g','LineWidth',2, 'DisplayName', 'Actual trajectory');

legend(mapPlotAxes, 'Location',  'northeast', ...
                'TextColor', [1 1 1], 'FontWeight', 'bold');
drawnow limitrate
end