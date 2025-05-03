%helperTrackLastKeyFrameKLT Estimate the camera pose by tracking the last key frame
%   [currPose, mapPointIdx, featureIdx] = helperTrackLastKeyFrameKLT(tracker,
%   currI, mapPoints, views, currFeatures, currPoints, lastKeyFrameId, 
%   intrinsics) estimates the camera pose of the current frame by matching 
%   features with the previous key frame.
%
%   This is an example helper function that is subject to change or removal 
%   in future releases.
%
%   Inputs
%   ------
%   tracker           - A vision.PointTracker object
%   currI             - Current image frame
%   mapPoints         - A mapPointSet objects storing map points
%   views             - View attributes of key frames
%   currFeatures      - Features in the current frame 
%   currPoints        - Feature points in the current frame
%   centerPoints      - Predicted locations of feature points from last key
%                       frame on the current frame
%   lastKeyFrameId    - ViewId of the last key frame 
%   intrinsics        - Camera intrinsics 
%   scaleFactor       - scale factor of features
%   
%   Outputs
%   -------
%   currPose          - Estimated camera pose of the current frame
%   mapPointIdx       - Indices of map points observed in the current frame
%   featureIdx        - Indices of features corresponding to mapPointIdx

%   Copyright 2024 The MathWorks, Inc.

function [currPose, mapPointIdx, featureIdx] = helperTrackLastKeyFrameKLT(...
    tracker, currI, mapPoints, views, currFeatures, currPoints, ...
    lastKeyFrameId, intrinsics)

% Match features from the previous key frame with known world locations
[index3d, index2d]    = findWorldPointsInView(mapPoints, lastKeyFrameId);
if isempty(coder.target())
    index3dVal = index3d;
    index2dVec = index2d;
else
    index3dVal = index3d{1};
    index2dVec = index2d{1};
end
lastKeyFrameFeatures  = views.Features{lastKeyFrameId}(index2dVec,:);

[centerPoints, validity] = tracker(currI);

searchRadius = 4; 

indexPairs   = matchFeaturesInRadius(binaryFeatures(lastKeyFrameFeatures(validity,:)), ...
    binaryFeatures(currFeatures.Features), currPoints, centerPoints(validity,:), searchRadius, ...
    'MatchThreshold', 20, 'MaxRatio', 0.9, 'Unique', true);

if size(indexPairs, 1) < 20
    indexPairs   = matchFeaturesInRadius(binaryFeatures(lastKeyFrameFeatures(validity,:)), ...
        binaryFeatures(currFeatures.Features), currPoints, centerPoints(validity,:), 2*searchRadius, ...
        'MatchThreshold', 20, 'MaxRatio', 0.9, 'Unique', true);
end

if size(indexPairs, 1) < 10
    pose        = rigidtform3d;
    currPose    = repmat(pose, 0, 0);
    mapPointIdx = zeros(0, 1);
    featureIdx  = zeros(0, 1, class(indexPairs));
    return
end


% Obtain the index of matched map points and features
tempIdx            = find(validity); % Convert to linear index
coder.varsize('mapPointIdx', [inf, 1], [1, 0]);
mapPointIdx        = index3dVal(tempIdx(indexPairs(:,1)));
coder.varsize('featureIdx', [inf, 1], [1, 0]);
featureIdx         = indexPairs(:,2);

matchedImagePoints = currPoints.Location(featureIdx,:);
matchedWorldPoints = mapPoints.WorldPoints(mapPointIdx, :);

% Estimate the camera pose
matchedImagePoints = cast(matchedImagePoints, 'like', matchedWorldPoints);
[currPose, inlier] = estworldpose(...
    matchedImagePoints, matchedWorldPoints, intrinsics, ...
    'Confidence', 95, 'MaxReprojectionError', 3, 'MaxNumTrials', 1e4);

% Refine the camera pose
currPose = bundleAdjustmentMotion(matchedWorldPoints(inlier,:), matchedImagePoints(inlier,:), ...
    currPose, intrinsics, 'PointsUndistorted', true, 'AbsoluteTolerance', 1e-7,...
    'RelativeTolerance', 1e-15, 'MaxIterations', 20);

mapPointIdx = mapPointIdx(inlier);
featureIdx = featureIdx(inlier);
end