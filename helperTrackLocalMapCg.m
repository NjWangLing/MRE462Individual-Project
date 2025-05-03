function [localKeyFrameIds, currPose, mapPointIdx, featureIdx, isKeyFrame, numPointsRefKeyFrame, localPointsIndices, localKeyFrameIdsInternal] = ...
    helperTrackLocalMapCg(mapPoints, vSetKeyFrames, mapPointIdx, ...
    featureIdx, currPose, currFeatures, currPoints, intrinsics, scaleFactor, numLevels, ...
    newKeyFrameAdded, lastKeyFrameIndex, currFrameIndex, numSkipFrames, numPointsKeyFrame, numPointsRefKeyFrame, localPointsIndices, localKeyFrameIdsInternal)
%helperTrackLocalMapCg Refine camera pose by tracking the local map
%
%   This is an example helper function that is subject to change or removal 
%   in future releases.
%
%   Inputs
%   ------
%   mapPoints                - A worldpointset object storing map points
%   vSetKeyFrames            - An imageviewset storing key frames
%   mapPointsIndices         - Indices of map points observed in the current frame
%   featureIndices           - Indices of features in the current frame 
%                              corresponding to map points denoted by mapPointsIndices                      
%   currPose                 - Current camera pose
%   currFeatures             - ORB Features in the current frame 
%   currPoints               - Feature points in the current frame
%   intrinsics               - Camera intrinsics 
%   scaleFactor              - scale factor of features
%   numLevels                - number of levels in feature exatraction
%   newKeyFrameAdded         - A boolean scalar indicating if a new key frame is
%                              added recently
%   lastKeyFrameIndex        - Frame index of the last key frame
%   currFrameIndex           - Frame index of the current frame
%   numSkipFrames            - Largest number of frames to skip
%   numPointsKeyFrame        - Minimum number of points tracked by a key frame
%   numPointsRefKeyFrame     - Number of points in reference key frame
%   localPointsIndices       - Indices of points in the local frame
%   localKeyFrameIdsInternal - View ids of the frames connected with
%                              current frame
%
%   Outputs
%   -------
%   localKeyFrameIds         - ViewIds of the local key frames 
%   currPose                 - Refined camera pose of the current frame
%   mapPointIdx              - Indices of map points observed in the current frame
%   featureIdx               - Indices of features in the current frame corresponding
%                              to mapPointIdx
%   isKeyFrame               - A boolean scalar indicating if the current frame is
%                              a key frame
%   numPointsKeyFrame        - Minimum number of points tracked by a key frame
%   numPointsRefKeyFrame     - Number of points in reference key frame
%   localPointsIndices       - Indices of points in the local frame
%   localKeyFrameIdsInternal - View ids of the frames connected with
%                              current frame

%   Copyright 2023 The MathWorks, Inc.
%#codegen

if isempty(numPointsRefKeyFrame) || newKeyFrameAdded
    [localPointsIndices, localKeyFrameIdsInternal, numPointsRefKeyFrame] = ...
    updateRefKeyFrameAndLocalPoints(mapPoints, vSetKeyFrames, mapPointIdx);
end
% Project the map into the frame and search for more map point correspondences
newMapPointIdx = setdiff(localPointsIndices, mapPointIdx, 'stable');
views = vSetKeyFrames.Views;
localFeatures  = getFeatures(mapPoints, views, newMapPointIdx);
[projectedPoints, inlierIndex, predictedScales, viewAngles] = removeOutlierMapPoints(mapPoints, ...
    currPose, intrinsics, newMapPointIdx, scaleFactor, numLevels);
newMapPointIdx = newMapPointIdx(inlierIndex);
localFeatures  = localFeatures(inlierIndex,:);

featureVector = uint32((1:size( currFeatures.Features, 1)).');
unmatchedfeatureIdx = setdiff(featureVector, ...
    featureIdx,'stable');
unmatchedFeatures   = currFeatures.Features(unmatchedfeatureIdx, :);
unmatchedValidPoints = currPoints.Location(unmatchedfeatureIdx, :);
% Search radius depends on scale and view direction
searchRadius    = 4*ones(size(localFeatures, 1), 1);
searchRadius(viewAngles<3) = 2.5;
searchRadius    = searchRadius.*predictedScales;
indexPairs = matchFeaturesInRadius(binaryFeatures(localFeatures),...
    binaryFeatures(unmatchedFeatures), unmatchedValidPoints, projectedPoints, ...
    searchRadius, 'MatchThreshold', 40, 'MaxRatio', 0.9, 'Unique', true);
% Filter by scales
indexPairs1 = indexPairs(:, 1);
currScale = currPoints.Scale(indexPairs(:, 2));
minScales = max(1, predictedScales(indexPairs1)/scaleFactor);
maxScales = predictedScales(indexPairs1);
miniScale = coder.nullcopy(zeros(size(minScales), 'logical'));
maxiScale = coder.nullcopy(zeros(size(maxScales), 'logical'));
parfor i=1:size(minScales, 1)
    if (currScale(i, 1)>=(minScales(i,1)))
        miniScale(i) = 1;
    else
        miniScale(i) = 0;
    end

    if (currScale(i,1)<=maxScales(i,1))
        maxiScale(i) = 1;
    else
        maxiScale(i) = 0;
    end
end
isGoodScale = coder.nullcopy(zeros(size(minScales), 'logical'));
% Filter by scales
parfor i=1:size(minScales, 1)
    isGoodScale(i) = miniScale(i) & maxiScale(i);
end
indexPairs  = indexPairs(isGoodScale, :);
% Refine camera pose with more 3D-to-2D correspondences
mapPointIdx   = [newMapPointIdx(indexPairs(:,1)); mapPointIdx];
featureIdx     = [unmatchedfeatureIdx(indexPairs(:,2)); featureIdx];
matchedMapPoints   = mapPoints.WorldPoints(mapPointIdx,:);
matchedImagePoints = currPoints.Location(featureIdx,:);
isKeyFrame = checkKeyFrame(numPointsRefKeyFrame, lastKeyFrameIndex, ...
    currFrameIndex, mapPointIdx, numSkipFrames, numPointsKeyFrame);

localKeyFrameIds = localKeyFrameIdsInternal;

if isKeyFrame
    % Refine camera pose only if the current frame is a key frame
    currPose = bundleAdjustmentMotion(matchedMapPoints, matchedImagePoints, ...
        currPose, intrinsics, 'PointsUndistorted', true, ...
        'AbsoluteTolerance', 1e-7, 'RelativeTolerance', 1e-16,'MaxIterations', 20);
end
end

function [localPointsIndices, localKeyFrameIds, numPointsRefKeyFrame] = ...
    updateRefKeyFrameAndLocalPoints(mapPoints, vSetKeyFrames, pointIndices)

if vSetKeyFrames.NumViews == 1
    localKeyFrameIds = vSetKeyFrames.Views.ViewId;
    localPointsIndices = (1:mapPoints.Count)';
    numPointsRefKeyFrame = mapPoints.Count;
    return
end

% The reference key frame has the most covisible map points 
viewIds = findViewsOfWorldPoint(mapPoints, pointIndices);
colSize = size(viewIds{1}, 2);
totalSize = 0;
for i=1:size(viewIds, 1)
    totalSize = totalSize+size(viewIds{i}, 1);
end
viewIdClass = class(viewIds{1});
viewIdsCol = coder.nullcopy(zeros(totalSize, colSize, viewIdClass));
ind = 1;
for i=1:size(viewIds, 1)
    currSize = size(viewIds{i}, 1);
    viewIdsCol(ind:ind+currSize-1) = viewIds{i};
    ind = ind+currSize;
end
refKeyFrameId = mode(viewIdsCol);
localKeyFrames = connectedViews(vSetKeyFrames, refKeyFrameId, "MaxDistance", 2);
localKeyFrameIds = [localKeyFrames.ViewId; refKeyFrameId];
pointIdx = findWorldPointsInView(mapPoints, localKeyFrameIds);
coder.varsize('keyFrameIdx', [inf, 1], [1, 0]);
keyFrameIdx = zeros(size(localKeyFrameIds, 1), 1, 'logical');
for i=1:size(localKeyFrameIds, 1)
    if(localKeyFrameIds(i) == refKeyFrameId)
        keyFrameIdx(i, 1) = true;
    end
end
totalSize = 0;
for i=1:size(pointIdx, 1)
    totalSize = totalSize+size(pointIdx{i}, 1);
end
pointIdxCol = coder.nullcopy(zeros(totalSize, 1));
ind = 1;
for i=1:size(pointIdx, 1)
    currSize = size(pointIdx{i}, 1);
    pointIdxCol(ind:ind+currSize-1) = pointIdx{i};
    ind = ind+currSize;
end
numPointsRefKeyFrame = 0;
if numel(pointIdx) > 1
    for i=1:size(keyFrameIdx, 1)
        if keyFrameIdx(i) == true
            numPointsRefKeyFrame = numPointsRefKeyFrame+ numel(pointIdx{i});
        end
    end
    localPointsIndices = sort(pointIdxCol);
else
    numPointsRefKeyFrame = numel(pointIdx{1});
    pointIdxCol = pointIdx{1};
    localPointsIndices = sort(pointIdxCol);
end
end

function features = getFeatures(mapPoints, views, mapPointIdx)

% Efficiently retrieve features and image points corresponding to map points
% denoted by mapPointIdx
allIndices = zeros(1, numel(mapPointIdx));

% ViewId and offset pair
count = zeros(coder.ignoreConst(0), 2); % (ViewId, NumFeatures)
viewsFeatures = views.Features;

for i = 1:numel(mapPointIdx)
    index3d  = mapPointIdx(i);
    
    viewId   = double(mapPoints.RepresentativeViewId(index3d));
    
    if isempty(count)
        count = [viewId, size(viewsFeatures{viewId},1)];
    elseif ~any(count(:,1) == viewId)
        count = [count; viewId, size(viewsFeatures{viewId},1)];
    end
    
    idx = find(count(:,1)==viewId);
    if idx > 1
        offset = sum(count(1:idx(1)-1,2));
    else
        offset = 0;
    end
    allIndices(i) = mapPoints.RepresentativeFeatureIndex(index3d) + offset;
end

uIds = count(:,1);

% Concatenating features and indexing once is faster than accessing via a for loop
totalSize = 0;
colSize = size(viewsFeatures{uIds(1)}, 2);
for i=1:size(uIds, 1)
    totalSize = totalSize+size(viewsFeatures{uIds(i)}, 1);
end
featuresClass = class(viewsFeatures{uIds(1)});
allFeatures = coder.nullcopy(zeros(totalSize, colSize, featuresClass));
ind = 1;
for i=1:size(uIds, 1)
    curSize = size(viewsFeatures{uIds(i)}, 1);
    allFeatures(ind:ind+curSize-1, :) = viewsFeatures{uIds(i)};
    ind = ind+curSize;
end
features    = allFeatures(allIndices, :);
end

function [projectedPoints, inliers, predictedScales, viewAngles] = removeOutlierMapPoints(...
    mapPoints, pose, intrinsics, localPointsIndices, scaleFactor, ...
    numLevels)

% 1) Points within the image bounds
xyzPoints = mapPoints.WorldPoints(localPointsIndices, :);
[projectedPoints, isInImage] = world2img(xyzPoints, pose2extr(pose), intrinsics);

if isempty(projectedPoints)
    error('Tracking failed. Try inserting new key frames more frequently.')
end

% 2) Parallax less than 60 degrees
cameraNormVector = [0 0 1] * pose.Rotation;
cameraToPoints   = xyzPoints - pose.Translation;
viewDirection    = mapPoints.ViewingDirection(localPointsIndices, :);
validByView      = sum(viewDirection.*cameraToPoints, 2) > ...
    cosd(60)*(vecnorm(cameraToPoints, 2, 2));

% 3) Distance from map point to camera center is in the range of scale
% invariant depth
minDist          = mapPoints.DistanceLimits(localPointsIndices,1)/scaleFactor;
maxDist          = mapPoints.DistanceLimits(localPointsIndices,2)*scaleFactor;
dist             = vecnorm(xyzPoints - pose.Translation, 2, 2);

validByDistance  = dist > minDist & dist < maxDist;
inliers          = isInImage & validByView & validByDistance;
% Predicted scales
level= ceil(log(maxDist ./ dist)./log(scaleFactor));
level(level<0)   = 0;
level(level>=numLevels-1) = numLevels-1;
predictedScales = zeros(size(level), 'single');
for i=1:numel(level)
    temp = double(1);
    for j=1:level(i)
        temp = temp*double(scaleFactor);
    end
    predictedScales(i) = temp;
end
% View angles
viewAngles       = acosd(complex(sum(cameraNormVector.*cameraToPoints, 2) ./ ...
    vecnorm(cameraToPoints, 2, 2)));

predictedScales  = predictedScales(inliers);
viewAngles       = viewAngles(inliers);

projectedPoints = projectedPoints(inliers, :);
end

function isKeyFrame = checkKeyFrame(numPointsRefKeyFrame, lastKeyFrameIndex, ...
    currFrameIndex, mapPointsIndices, numSkipFrames, numPointsKeyFrame)

% More than numSkipFrames frames have passed from last key frame insertion
tooManyNonKeyFrames = currFrameIndex > lastKeyFrameIndex + numSkipFrames;

% Track less than numPointsKeyFrame map points
tooFewMapPoints     = numel(mapPointsIndices) < numPointsKeyFrame;

% Tracked map points are fewer than 90% of points tracked by
% the reference key frame
tooFewTrackedPoints = numel(mapPointsIndices) < 0.9 * numPointsRefKeyFrame;

isKeyFrame = (tooManyNonKeyFrames || tooFewMapPoints) && tooFewTrackedPoints;
end