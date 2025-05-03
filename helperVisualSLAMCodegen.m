function out = helperVisualSLAMCodegen(imagesCell)

% Set random seed for reproducibility
rng(0);
currFrameIdx = 1;
currI = imagesCell{currFrameIdx};
% Create a cameraIntrinsics object to store the camera intrinsic parameters.
% The intrinsics for the dataset can be found at the following page:
% https://vision.in.tum.de/data/datasets/rgbd-dataset/file_formats
% Note that the images in the dataset are already undistorted, hence there
% is no need to specify the distortion coefficients.
focalLength    = [535.4, 539.2];    % in units of pixels
principalPoint = [320.1, 247.6];    % in units of pixels
imageSize      = size(currI,[1 2]);  % in units of pixels
intrinsics     = cameraIntrinsics(focalLength, principalPoint, imageSize);

% Detect and extract ORB features
scaleFactor = 1.2;
numLevels   = 8;
numPoints   = 1000;
[preFeatures, prePoints] = helperDetectAndExtractFeatures(currI, scaleFactor, numLevels, numPoints);

currFrameIdx = currFrameIdx + 1;
firstI       = currI; % Preserve the first frame

isMapInitialized  = false;
relPose = rigidtform3d(eye(4, 4, class(prePoints.Location)));
xyzWorldPoints = zeros(coder.ignoreConst(0), 3, class(prePoints.Location));
currPoints = ORBPoints(zeros(coder.ignoreConst(0), 2));
currFeatures = binaryFeatures(zeros(coder.ignoreConst(0), 2, 'uint8'));
indexPairs = zeros(coder.ignoreConst(0), 2, 'uint32');
% Map initialization loop
while (isMapInitialized~=true && currFrameIdx < numel(imagesCell))
    currI = imagesCell{currFrameIdx};

    [currFeatures, currPoints] = helperDetectAndExtractFeatures(currI, scaleFactor, numLevels, numPoints);

    currFrameIdx = currFrameIdx + 1;

    % Find putative feature matches
    indexPairs = matchFeatures(preFeatures, currFeatures, 'Unique', true, ...
        'MaxRatio', 0.9, 'MatchThreshold', 40);

    preMatchedPoints = prePoints.Location(indexPairs(:,1), :);
    currMatchedPoints = currPoints.Location(indexPairs(:,2), :);

    % If not enough matches are found, check the next frame
    minMatches = 100;
    if size(indexPairs, 1) < minMatches
        continue
    end

    % Compute homography and evaluate reconstruction
    [tformH, scoreH, inliersIdxH] = helperComputeHomography(preMatchedPoints, currMatchedPoints);

    % Compute fundamental matrix and evaluate reconstruction
    [tformF, scoreF, inliersIdxF] = helperComputeFundamentalMatrix(preMatchedPoints, currMatchedPoints, intrinsics);

    % Select the model based on a heuristic
    ratio = scoreH/(scoreH + scoreF);
    ratioThreshold = 0.45;

    if ratio > ratioThreshold
        inlierTformIdx = inliersIdxH;
        % Computes the camera location up to scale. Use half of the
        % points to reduce computation
        inlierPrePoints = preMatchedPoints(inlierTformIdx(1:2:end), :);
        inlierCurrPoints = currMatchedPoints(inlierTformIdx(1:2:end), :);
        [relPose, validFraction] = estrelpose(tformH, intrinsics, ...
            inlierPrePoints, inlierCurrPoints);
    else
        inlierTformIdx = inliersIdxF;
        % Computes the camera location up to scale. Use half of the
        % points to reduce computation
        inlierPrePoints = preMatchedPoints(inlierTformIdx(1:2:end), :);
        inlierCurrPoints = currMatchedPoints(inlierTformIdx(1:2:end), :);
        [relPose, validFraction] = estrelpose(tformF, intrinsics, ...
            inlierPrePoints, inlierCurrPoints);
    end

    if validFraction < 0.9 || numel(relPose)==3
        continue
    end

    % Triangulate two views to obtain 3-D map points
    minParallax = 1; % In degrees
    iPrePoints = preMatchedPoints(inlierTformIdx(:), :);
    iCurPoints = currMatchedPoints(inlierTformIdx(:), :);
    [isValid, xyzWorldPoints, inlierTriangulationIdx] = helperTriangulateTwoFrames(...
        rigidtform3d, relPose, iPrePoints, iCurPoints, intrinsics, minParallax);

    if ~isValid
        continue
    end

    % Get the original index of features in the two key frames
    indexPairs = indexPairs(inlierTformIdx(inlierTriangulationIdx),:);

    isMapInitialized = true;
end % End of map initialization
if isMapInitialized==false
    error('Unable to initialize the map.');
end
vSetKeyFrames = imageviewset;

% Create an empty worldpointset object to store 3-D map points
mapPointSet   = worldpointset;

% Add the first key frame. Place the camera associated with the first
% key frame at the origin, oriented along the Z-axis
preViewId     = 1;
vSetKeyFrames = addView(vSetKeyFrames, preViewId, rigidtform3d(eye(4, 4, class(relPose.T))), 'Points', prePoints,...
    'Features', preFeatures.Features);

% Add the second key frame
currViewId    = 2;
vSetKeyFrames = addView(vSetKeyFrames, currViewId, relPose, Points=currPoints,...
    Features=currFeatures.Features);

% Add connection between the first and the second key frame
vSetKeyFrames = addConnection(vSetKeyFrames, preViewId, currViewId, relPose, Matches=indexPairs);

% Add 3-D map points
[mapPointSet, newPointIdx] = addWorldPoints(mapPointSet, xyzWorldPoints);
% Add observations of the map points
% Add image points corresponding to the map points in the first key frame
mapPointSet   = addCorrespondences(mapPointSet, preViewId, newPointIdx, indexPairs(:,1));

% Add image points corresponding to the map points in the second key frame
mapPointSet   = addCorrespondences(mapPointSet, currViewId, newPointIdx, indexPairs(:,2));

bofData         = coder.load('bagOfFeaturesDataSLAMCg.mat');
bagObj = bofData.bag;
bagObj.CustomExtractor = @helperORBFeatureExtractorFunction;
bagObj.Encoder.EncoderFunction = @vision.internal.bof.EncoderBinaryFeatures;
bagObj.UsingCustomExtractor = coder.const(true);
bof = bagOfFeatures.loadobj(bagObj);
% Initialize the place recognition database
loopDatabase    = invertedImageIndex(bof,SaveFeatureLocations=false);

% Add features of the first two key frames to the database
addImageFeatures(loopDatabase, preFeatures, preViewId);
addImageFeatures(loopDatabase, currFeatures, currViewId);
% 
% Run full bundle adjustment on the first two key frames
tracks       = findTracks(vSetKeyFrames);
cameraPoses  = poses(vSetKeyFrames);
% Create poses table
aPoses = coder.nullcopy(cell(size(cameraPoses.ViewId, 1), 1));
for i=1:size(cameraPoses.ViewId, 1)
    aPoses{i} = cameraPoses.AbsolutePose(i);
end
cameraPosesTable = table(cameraPoses.ViewId, aPoses, 'VariableNames', {'ViewId', 'AbsolutePose'} );
[refinedPoints, refinedAbsPoses] = bundleAdjustment(xyzWorldPoints, tracks, ...
    cameraPosesTable, intrinsics, FixedViewIDs=1, ...
    PointsUndistorted=true, AbsoluteTolerance=1e-7,...
    RelativeTolerance=1e-15, MaxIterations=20, ...
    Solver="preconditioned-conjugate-gradient");
% Scale the map and the camera pose using the median depth of map points
medianDepth   = median(vecnorm(refinedPoints.'));
refinedPoints = refinedPoints / medianDepth;

refinedAbsPoses.AbsolutePose{currViewId}.Translation = ...
    (refinedAbsPoses.AbsolutePose{currViewId}.Translation) / medianDepth;
relPose.Translation = (relPose.Translation)/medianDepth;
% Update key frames with the refined poses
for i=1:size(refinedAbsPoses, 1)
    vSetKeyFrames = updateView(vSetKeyFrames, refinedAbsPoses.ViewId(i), refinedAbsPoses.AbsolutePose{i});
end
vSetKeyFrames = updateConnection(vSetKeyFrames, preViewId, currViewId, relPose);

% Update map points with the refined positions
mapPointSet = updateWorldPoints(mapPointSet, newPointIdx, refinedPoints);

% Update view direction and depth
mapPointSet = updateLimitsAndDirection(mapPointSet, newPointIdx, vSetKeyFrames.Views);

% % Update representative view
mapPointSet = updateRepresentativeView(mapPointSet, newPointIdx, vSetKeyFrames.Views);

% ViewId of the current key frame
currKeyFrameId   = currViewId;

% ViewId of the last key frame
lastKeyFrameId   = currViewId;

% Index of the last key frame in the input image sequence
lastKeyFrameIdx  = currFrameIdx - 1;

% Indices of all the key frames in the input image sequence
addedFramesIdx   = [1; lastKeyFrameIdx];

isLoopClosed     = false;

isLastFrameKeyFrame = true;
numPointsRefKeyFrame = 0;
coder.varsize('localPointsIndices', [inf, 1], [1, 0]);
localPointsIndices = zeros(0, 1);
coder.varsize('localKeyFrameIdsInternal', [inf, 1], [1, 0]);
localKeyFrameIdsInternal = zeros(0, 1, 'uint32');

% Create and initialize the KLT tracker
tracker = vision.PointTracker(MaxBidirectionalError = 5);
initialize(tracker, currPoints.Location(indexPairs(:,2), :), currI);

while currFrameIdx < numel(imagesCell)
    currI = imagesCell{currFrameIdx};
    [currFeatures, currPoints] = helperDetectAndExtractFeatures(currI, scaleFactor, numLevels, numPoints);
    % Track the last key frame
    [currPose, mapPointsIdx, featureIdx] = helperTrackLastKeyFrameKLT(tracker, currI, mapPointSet, ...
        vSetKeyFrames.Views, currFeatures, currPoints, lastKeyFrameId, intrinsics);

    % Track the local map and check if the current frame is a key frame.
    % A frame is a key frame if both of the following conditions are satisfied:
    %
    % 1. At least 20 frames have passed since the last key frame or the
    %    current frame tracks fewer than 100 map points.
    % 2. The map points tracked by the current frame are fewer than 90% of
    %    points tracked by the reference key frame.
    %
    % Tracking performance is sensitive to the value of numPointsKeyFrame.
    % If tracking is lost, try a larger value.
    %
    % localKeyFrameIds:   ViewId of the connected key frames of the current frame
    numSkipFrames     = 20;
    numPointsKeyFrame = 80;
    [localKeyFrameIds, currPose, mapPointsIdx, featureIdx, isKeyFrame, numPointsRefKeyFrame, localPointsIndices, localKeyFrameIdsInternal] = ...
        helperTrackLocalMapCg(mapPointSet, vSetKeyFrames, mapPointsIdx, ...
        featureIdx, currPose, currFeatures, currPoints, intrinsics, scaleFactor, numLevels, ...
        isLastFrameKeyFrame, lastKeyFrameIdx, currFrameIdx, numSkipFrames, numPointsKeyFrame, numPointsRefKeyFrame, localPointsIndices, localKeyFrameIdsInternal);
    
    if ~isKeyFrame
        currFrameIdx        = currFrameIdx + 1;
        isLastFrameKeyFrame = false;
        continue
    else
        isLastFrameKeyFrame = true;
    end

    % Update current key frame ID
    currKeyFrameId  = currKeyFrameId + 1;
    curPoseSingle = rigidtform3d(single(currPose.A));
    % Add the new key frame
    [mapPointSet, vSetKeyFrames] = helperAddNewKeyFrame(mapPointSet, vSetKeyFrames, ...
        curPoseSingle, currFeatures, currPoints, mapPointsIdx, featureIdx, localKeyFrameIds);
    % Remove outlier map points that are observed in fewer than 3 key frames
    mapPointSet = helperCullRecentMapPoints(mapPointSet, mapPointsIdx, newPointIdx);
    % Create new map points by triangulation
    minNumMatches = 10;
    minParallax   = 3;
    [mapPointSet, vSetKeyFrames, newPointIdx] = helperCreateNewMapPoints(mapPointSet, vSetKeyFrames, ...
        currKeyFrameId, intrinsics, scaleFactor, minNumMatches, minParallax);

    % Local bundle adjustment
    [refinedViews, dist] = connectedViews(vSetKeyFrames, currKeyFrameId, MaxDistance=2);
    refinedKeyFrameIds = refinedViews.ViewId;
    fixedViewIds = refinedKeyFrameIds(dist==2);
    fixedViewIds = fixedViewIds(1:min(10, numel(fixedViewIds)));
    % Refine local key frames and map points
    [mapPointSet, vSetKeyFrames, mapPointIdx] = bundleAdjustment(...
        mapPointSet, vSetKeyFrames, [refinedKeyFrameIds; currKeyFrameId], intrinsics, ...
        FixedViewIDs=fixedViewIds, PointsUndistorted=true, AbsoluteTolerance=1e-7,...
        RelativeTolerance=1e-16, Solver="preconditioned-conjugate-gradient", ...
        MaxIterations=10);

    mapPointSet = updateLimitsAndDirection(mapPointSet, mapPointIdx, vSetKeyFrames.Views);

    % Update representative view
    mapPointSet = updateRepresentativeView(mapPointSet, mapPointIdx, vSetKeyFrames.Views);

    % Set the feature points to be tracked
    [~, index2d] = findWorldPointsInView(mapPointSet, currKeyFrameId);
    setPoints(tracker, currPoints.Location(index2d{1}, :));
    
    % Check loop closure after some key frames have been created
    if currKeyFrameId > 20
        % Minimum number of feature matches of loop edges
        loopEdgeNumMatches = 50;

        % Detect possible loop closure key frame candidates
        [isDetected, validLoopCandidates] = helperCheckLoopClosureCg(vSetKeyFrames, currKeyFrameId, ...
            loopDatabase, currI, loopEdgeNumMatches);

        if isDetected
            % Add loop closure connections
            [isLoopClosed, mapPointSet, vSetKeyFrames] = helperAddLoopConnections(...
                mapPointSet, vSetKeyFrames, validLoopCandidates, currKeyFrameId, ...
                currFeatures, loopEdgeNumMatches);
        end
    end

    % If no loop closure is detected, add current features into the database
    if ~isLoopClosed
        addImageFeatures(loopDatabase,  currFeatures, currKeyFrameId);
    end

    % Update IDs and indices
    lastKeyFrameId  = currKeyFrameId;
    lastKeyFrameIdx = currFrameIdx;
    addedFramesIdx  = [addedFramesIdx; currFrameIdx]; %#ok<AGROW>
    currFrameIdx    = currFrameIdx + 1;
end % End of main loop

%% Commenting for time being
% Optimize the poses
minNumMatches      = 30;
vSetKeyFramesOptim = optimizePoses(vSetKeyFrames, minNumMatches, Tolerance=1e-16);
% Plot the optimized camera trajectory
optimizedPoses  = poses(vSetKeyFramesOptim);
optimizedPose = coder.nullcopy(cell(size(optimizedPoses.ViewId, 1), 1));
for i=1:size(optimizedPoses.ViewId, 1)
    optimizedPose{i} = optimizedPoses.AbsolutePose(i).A;
end

%% This code is partially used.These values are used to visualize mono slam
% results
out.viewIds = vSetKeyFrames.Views.ViewId;

camPoses    = poses(vSetKeyFrames);
currPose    = camPoses(end,:); % Contains both ViewId and Pose

% Ensure the rotation matrix is a rigid transformation
R = double(currPose.AbsolutePose.R);
t = double(currPose.AbsolutePose.Translation);
[U, ~, V] = svd(R);
currPose.AbsolutePose.A = eye(4, 4, class(currPose.AbsolutePose.A));
currPose.AbsolutePose.A(1:3, 4) = t;
currPose.AbsolutePose.A(1:3, 1:3) = U * V';

xyzPoints   = mapPointSet.WorldPoints;

XLim = [-1.5 1.5];
YLim = [-1 0.5];
ZLim = [-0.5 2];

% Only plot the points within the limit
xyzPoints1 = xyzPoints(:, 1);
xyzPoints2 = xyzPoints(:, 2);
xyzPoints3 = xyzPoints(:, 3);
inPlotRange = xyzPoints1 > XLim(1) & ...
    xyzPoints1 < XLim(2) & xyzPoints2 > YLim(1) & ...
    xyzPoints2 < YLim(2) & xyzPoints3 > ZLim(1) & ...
    xyzPoints3 < ZLim(2);
xyzPoints   = xyzPoints(inPlotRange, :);

out.xyzPoints = xyzPoints;
pose = coder.nullcopy(cell(size(camPoses.ViewId, 1), 1));
for i=1:size(camPoses.ViewId, 1)
pose{i} = camPoses.AbsolutePose(i).A;
end
out.optimizedPoses = optimizedPose;
out.estimatedPoses = pose;
out.vId = camPoses.ViewId;
out.addedFramesIdx = addedFramesIdx;
end

%--------------------------------------------------------------------------
function mapPointSet = helperCullRecentMapPoints(mapPointSet, mapPointsIdx, newPointIdx)
outlierIdx    = setdiff(newPointIdx, mapPointsIdx, 'stable');
if ~isempty(outlierIdx)
    mapPointSet   = removeWorldPoints(mapPointSet, outlierIdx);
end
end

%--------------------------------------------------------------------------
function [H, score, inliersIndex] = helperComputeHomography(matchedPoints1, matchedPoints2)

[H, inliersLogicalIndex] = estgeotform2d( ...
    matchedPoints1, matchedPoints2, 'projective', ...
    'MaxNumTrials', 1e3, 'MaxDistance', 4, 'Confidence', 90);

inliersIndex  = find(inliersLogicalIndex);
locations1 = matchedPoints1(inliersLogicalIndex,:);
locations2 = matchedPoints2(inliersLogicalIndex,:);
xy1In2     = transformPointsForward(H, locations1);
xy2In1     = transformPointsInverse(H, locations2);
error1in2  = sum((locations2 - xy1In2).^2, 2);
error2in1  = sum((locations1 - xy2In1).^2, 2);

outlierThreshold = 6;

score = sum(max(outlierThreshold - error1in2, 0)) + ...
    sum(max(outlierThreshold - error2in1, 0));
end

%--------------------------------------------------------------------------
function [FActual, scoreOut, inliersIndex] = helperComputeFundamentalMatrix(matchedPoints1, matchedPoints2,intrinsics)

[E, inliersLogicalIndex] = estimateEssentialMatrix( ...
    matchedPoints1, matchedPoints2, intrinsics, MaxNumTrials=1e3, MaxDistance=4);

F = intrinsics.K'\ E /intrinsics.K;

locations1 = matchedPoints1(inliersLogicalIndex,:);
locations2 = matchedPoints2(inliersLogicalIndex,:);

inliersIndex  = find(inliersLogicalIndex);

% Distance from points to epipolar line
lineIn1   = epipolarLine(F', locations2);
updatedLocations1 = [locations1, ones(size(locations1, 1),1)];
sumLocations1 = sum(updatedLocations1.* lineIn1, 2);
error2in1 = (sumLocations1).^2 ...
    ./ sum(lineIn1(:,1:2).^2, 2);
lineIn2   = epipolarLine(F, locations1);
updatedLocations2 = [locations2, ones(size(locations2, 1),1)];
sumLocations2 = sum(updatedLocations2.* lineIn2, 2);
error1in2 = (sumLocations2).^2 ...
    ./ sum(lineIn2(:,1:2).^2, 2);

outlierThreshold = 4;

score1 = sum(max(outlierThreshold-error1in2, 0)) + ...
    sum(max(outlierThreshold-error2in1, 0));
scoreOut = cast(score1, class(locations2));
FActual = cast(F, class(locations2));
end

%--------------------------------------------------------------------------
function [isValid, xyzPoints, inlierIdx] = helperTriangulateTwoFrames(...
    pose1, pose2, matchedPoints1, matchedPoints2, intrinsics, minParallax)

camMatrix1 = cameraProjection(intrinsics, pose2extr(pose1));
camMatrix2 = cameraProjection(intrinsics, pose2extr(pose2));
[xyzPoints, reprojectionErrors, isInFront] = triangulate(matchedPoints1, ...
    matchedPoints2, camMatrix1, camMatrix2);

% Filter points by view direction and reprojection error
minReprojError = 1;
inlierIdx  = isInFront & reprojectionErrors < minReprojError;
xyzPoints  = xyzPoints(inlierIdx ,:);

% A good two-view with significant parallax
ray1       = xyzPoints - pose1.Translation;
ray2       = xyzPoints - pose2.Translation;
cosAngle   = sum(ray1 .* ray2, 2) ./ (vecnorm(ray1, 2, 2) .* vecnorm(ray2, 2, 2));

% Check parallax
isValid = all(cosAngle < cosd(minParallax) & cosAngle>0);
end

%--------------------------------------------------------------------------
function [features, validPoints] = helperDetectAndExtractFeatures(Irgb, ...
    scaleFactor, numLevels, numPoints, varargin)

% Detect ORB features
Igray  = im2gray(Irgb);

points = detectORBFeatures(Igray, ScaleFactor=scaleFactor, NumLevels=numLevels);

% Select a subset of features, uniformly distributed throughout the image
points = selectUniform(points, numPoints, size(Igray, 1:2));

% Extract features
[features, validPoints] = extractFeatures(Igray, points);
end