clc;
clear;
close all;
videoFile = "C:/Users/njwan/OneDrive/Desktop/visul-SLAM.mp4";
vReader = VideoReader(videoFile);
numFrames = floor(vReader.Duration * vReader.FrameRate);
imageArray = cell(1, numFrames);
for i = 1:numFrames
    imageArray{i} = readFrame(vReader);
end
tempImageFolder = fullfile(tempdir, 'visul_slam_frames');
if ~exist(tempImageFolder, "dir")
    mkdir(tempImageFolder);
end
if isempty(dir(fullfile(tempImageFolder, '*.png')))
    for i = 1:numFrames
        imwrite(imageArray{i}, fullfile(tempImageFolder, sprintf('frame_%04d.png', i)));
    end
end
imds = imageDatastore(tempImageFolder);
currFrameIdx = 1;
currI = readimage(imds, currFrameIdx);
himage = imshow(currI);
rng(0);
scaleFactor = 1920 / 640;
focalLength = [535.4, 539.2] * scaleFactor; 
principalPoint = [320.1, 247.6] * scaleFactor; 
imageSize = [1080, 1920];
intrinsics = cameraIntrinsics(focalLength, principalPoint, imageSize);
scaleFactor = 1.2;
numLevels   = 8;
numPoints   = 1000;
[preFeatures, prePoints] = helperDetectAndExtractFeatures(currI, scaleFactor, numLevels, numPoints); 

currFrameIdx = currFrameIdx + 1;
firstI       = currI;

isMapInitialized  = false;
while ~isMapInitialized && currFrameIdx < numel(imds.Files)
    currI = readimage(imds, currFrameIdx);

    [currFeatures, currPoints] = helperDetectAndExtractFeatures(currI, scaleFactor, numLevels, numPoints); 

    currFrameIdx = currFrameIdx + 1;
    indexPairs = matchFeatures(preFeatures, currFeatures, Unique=true, ...
        MaxRatio=0.9, MatchThreshold=40);
    minMatches = 100;
    if size(indexPairs, 1) < minMatches
        continue
    end
preMatchedPoints  = prePoints(indexPairs(:,1),:);
    currMatchedPoints = currPoints(indexPairs(:,2),:);
    [tformH, scoreH, inliersIdxH] = helperComputeHomography(preMatchedPoints, currMatchedPoints);
    [tformF, scoreF, inliersIdxF] = helperComputeFundamentalMatrix(preMatchedPoints, currMatchedPoints, intrinsics);
    ratio = scoreH/(scoreH + scoreF);
    ratioThreshold = 0.45;
    if ratio > ratioThreshold
        inlierTformIdx = inliersIdxH;
        tform          = tformH;
    else
        inlierTformIdx = inliersIdxF;
        tform          = tformF;
    end
    inlierPrePoints  = preMatchedPoints(inlierTformIdx);
    inlierCurrPoints = currMatchedPoints(inlierTformIdx);
    [relPose, validFraction] = estrelpose(tform, intrinsics, ...
        inlierPrePoints(1:2:end), inlierCurrPoints(1:2:end));
    if validFraction < 0.9 || numel(relPose)>1
        continue
    end
    minParallax = 1;
    [isValid, xyzWorldPoints, inlierTriangulationIdx] = helperTriangulateTwoFrames(...
        rigidtform3d, relPose, inlierPrePoints, inlierCurrPoints, intrinsics, minParallax);

    if ~isValid
        continue
    end
    indexPairs = indexPairs(inlierTformIdx(inlierTriangulationIdx),:);

    isMapInitialized = true;

    disp(['Map initialized with frame 1 and frame ', num2str(currFrameIdx-1)])
end 
if isMapInitialized
    close(himage.Parent.Parent); 
    hfeature = showMatchedFeatures(firstI, currI, prePoints(indexPairs(:,1)), ...
        currPoints(indexPairs(:, 2)), "Montage");
else
    error('Unable to initialize the map.')
end
vSetKeyFrames = imageviewset;
mapPointSet   = worldpointset;
preViewId     = 1;
vSetKeyFrames = addView(vSetKeyFrames, preViewId, rigidtform3d, Points=prePoints,...
    Features=preFeatures.Features);
currViewId    = 2;
vSetKeyFrames = addView(vSetKeyFrames, currViewId, relPose, Points=currPoints,...
    Features=currFeatures.Features);
vSetKeyFrames = addConnection(vSetKeyFrames, preViewId, currViewId, relPose, Matches=indexPairs);
[mapPointSet, newPointIdx] = addWorldPoints(mapPointSet, xyzWorldPoints);
mapPointSet   = addCorrespondences(mapPointSet, preViewId, newPointIdx, indexPairs(:,1));
mapPointSet   = addCorrespondences(mapPointSet, currViewId, newPointIdx, indexPairs(:,2));
bag  = bagOfFeaturesDBoW("bagOfFeatures.bin.gz");
loopDatabase    = dbowLoopDetector(bag);
addVisualFeatures(loopDatabase, preViewId, preFeatures);
addVisualFeatures(loopDatabase, currViewId, currFeatures);
tracks       = findTracks(vSetKeyFrames);
cameraPoses  = poses(vSetKeyFrames);
[refinedPoints, refinedAbsPoses] = bundleAdjustment(xyzWorldPoints, tracks, ...
    cameraPoses, intrinsics, FixedViewIDs=1, ...
    PointsUndistorted=true, AbsoluteTolerance=1e-7,...
    RelativeTolerance=1e-15, MaxIteration=20, ...
    Solver="preconditioned-conjugate-gradient");
medianDepth   = median(vecnorm(refinedPoints.'));
refinedPoints = refinedPoints / medianDepth;
refinedAbsPoses.AbsolutePose(currViewId).Translation = ...
    refinedAbsPoses.AbsolutePose(currViewId).Translation / medianDepth;
relPose.Translation = relPose.Translation/medianDepth;
vSetKeyFrames = updateView(vSetKeyFrames, refinedAbsPoses);
vSetKeyFrames = updateConnection(vSetKeyFrames, preViewId, currViewId, relPose);
mapPointSet = updateWorldPoints(mapPointSet, newPointIdx, refinedPoints); 
mapPointSet = updateLimitsAndDirection(mapPointSet, newPointIdx, vSetKeyFrames.Views);
mapPointSet = updateRepresentativeView(mapPointSet, newPointIdx, vSetKeyFrames.Views);
close(hfeature.Parent.Parent);
featurePlot   = helperVisualizeMatchedFeatures(currI, currPoints(indexPairs(:,2)));
mapPlot       = helperVisualizeMotionAndStructure(vSetKeyFrames, mapPointSet);
showLegend(mapPlot);
currKeyFrameId   = currViewId;
lastKeyFrameId   = currViewId;
lastKeyFrameIdx  = currFrameIdx - 1; 
addedFramesIdx   = [1; lastKeyFrameIdx];
isLoopClosed     = false;
isLastFrameKeyFrame = true;
tracker = vision.PointTracker(MaxBidirectionalError = 5);
initialize(tracker, currPoints.Location(indexPairs(:,2), :), currI);
while currFrameIdx < numel(imds.Files)  
    currI = readimage(imds, currFrameIdx);
    [currFeatures, currPoints] = helperDetectAndExtractFeatures(currI, scaleFactor, numLevels, numPoints);
    [currPose, mapPointsIdx, featureIdx] = helperTrackLastKeyFrameKLT(tracker, currI, mapPointSet, ...
        vSetKeyFrames.Views, currFeatures, currPoints, lastKeyFrameId, intrinsics);
    numSkipFrames     = 20;
    numPointsKeyFrame = 90;
    [localKeyFrameIds, currPose, mapPointsIdx, featureIdx, isKeyFrame] = ...
        helperTrackLocalMap(mapPointSet, vSetKeyFrames, mapPointsIdx, ...
        featureIdx, currPose, currFeatures, currPoints, intrinsics, scaleFactor, ...
        isLastFrameKeyFrame, lastKeyFrameIdx, currFrameIdx, numSkipFrames, numPointsKeyFrame);
    updatePlot(featurePlot, currI, currPoints(featureIdx));

    if ~isKeyFrame
        currFrameIdx        = currFrameIdx + 1;
        isLastFrameKeyFrame = false;
        continue
    else
        currKeyFrameId      = currKeyFrameId + 1;
        isLastFrameKeyFrame = true;
    end
    [mapPointSet, vSetKeyFrames] = helperAddNewKeyFrame(mapPointSet, vSetKeyFrames, ...
        currPose, currFeatures, currPoints, mapPointsIdx, featureIdx, localKeyFrameIds);
    outlierIdx    = setdiff(newPointIdx, mapPointsIdx);
    if ~isempty(outlierIdx)
        mapPointSet   = removeWorldPoints(mapPointSet, outlierIdx);
    end
    minNumMatches = 10;
    minParallax   = 3;
    [mapPointSet, vSetKeyFrames, newPointIdx] = helperCreateNewMapPoints(mapPointSet, vSetKeyFrames, ...
        currKeyFrameId, intrinsics, scaleFactor, minNumMatches, minParallax);
    [refinedViews, dist] = connectedViews(vSetKeyFrames, currKeyFrameId, MaxDistance=2);
    refinedKeyFrameIds = refinedViews.ViewId;
    fixedViewIds = refinedKeyFrameIds(dist==2);
    fixedViewIds = fixedViewIds(1:min(10, numel(fixedViewIds)));
    [mapPointSet, vSetKeyFrames, mapPointIdx] = bundleAdjustment(...
        mapPointSet, vSetKeyFrames, [refinedKeyFrameIds; currKeyFrameId], intrinsics, ...
        FixedViewIDs=fixedViewIds, PointsUndistorted=true, AbsoluteTolerance=1e-7,...
        RelativeTolerance=1e-16, Solver="preconditioned-conjugate-gradient", ...
        MaxIteration=10);
    mapPointSet = updateLimitsAndDirection(mapPointSet, mapPointIdx, vSetKeyFrames.Views);
    mapPointSet = updateRepresentativeView(mapPointSet, mapPointIdx, vSetKeyFrames.Views);
    updatePlot(mapPlot, vSetKeyFrames, mapPointSet);
    [~, index2d] = findWorldPointsInView(mapPointSet, currKeyFrameId);
    setPoints(tracker, currPoints.Location(index2d, :));  
    if currKeyFrameId > 20
        loopEdgeNumMatches = 50;
        [isDetected, validLoopCandidates] = helperCheckLoopClosure(vSetKeyFrames, currKeyFrameId, ...
            loopDatabase, currFeatures, loopEdgeNumMatches);
        if isDetected 
            [isLoopClosed, mapPointSet, vSetKeyFrames] = helperAddLoopConnections(...
                mapPointSet, vSetKeyFrames, validLoopCandidates, currKeyFrameId, ...
                currFeatures, loopEdgeNumMatches);
        end
    end
    if ~isLoopClosed
        addVisualFeatures(loopDatabase,  currKeyFrameId, currFeatures);
    end
    lastKeyFrameId  = currKeyFrameId;
    lastKeyFrameIdx = currFrameIdx;
    addedFramesIdx  = [addedFramesIdx; currFrameIdx]; %#ok<AGROW>
    currFrameIdx    = currFrameIdx + 1;
end 
if isLoopClosed
    minNumMatches      = 30;
    vSetKeyFramesOptim = optimizePoses(vSetKeyFrames, minNumMatches, Tolerance=1e-16);
    mapPointSet = helperUpdateGlobalMap(mapPointSet, vSetKeyFrames, vSetKeyFramesOptim);
    updatePlot(mapPlot, vSetKeyFrames, mapPointSet);
    optimizedPoses  = poses(vSetKeyFramesOptim);
    plotOptimizedTrajectory(mapPlot, optimizedPoses)
    showLegend(mapPlot);
end
if isLoopClosed
    gTruthData = load("orbslamGroundTruth.mat");
    gTruth     = gTruthData.gTruth;
    plotActualTrajectory(mapPlot, gTruth(addedFramesIdx), optimizedPoses);
    showLegend(mapPlot);
end
if isLoopClosed
    errorMetrics = compareTrajectories(optimizedPoses.AbsolutePose, gTruth(addedFramesIdx), AlignmentType="similarity");
    disp(['Absolute RMSE for key frame location (m): ', num2str(errorMetrics.AbsoluteRMSE(2))]);
    figure
    ax = plot(errorMetrics, "absolute-translation");
    view(ax, [2.70 -49.20]);
end
firstImage = im2gray(readimage(imds, 1));
imageSize = size(firstImage); 
imageType = coder.typeof(uint8(0), imageSize, [1, 1]); 
imageArrayType = coder.typeof({imageType}, [1, Inf]); 
cpuConfig = coder.config('mex');
cpuConfig.TargetLang = 'C++';
cpuConfig.EnableAutoExtrinsicCalls = true; 
codegen -config cpuConfig helperVisualSLAMCodegen -args {imageArrayType} -report
imageArray = cell(1, numel(imds.Files));
for i = 1:numel(imds.Files)
    imageArray{i} = im2gray(readimage(imds, i));
end
monoSlamOut = helperVisualSLAMCodegen_mex(imageArray);
mapPlot = helperVisualizeMonoVisualSlam(monoSlamOut);
function [features, validPoints] = helperDetectAndExtractFeatures(Irgb, ...
    scaleFactor, numLevels, numPoints, varargin)
Igray  = im2gray(Irgb);
points = detectORBFeatures(Igray, ScaleFactor=scaleFactor, NumLevels=numLevels);
points = selectUniform(points, numPoints, size(Igray, 1:2));
[features, validPoints] = extractFeatures(Igray, points);
end
function [H, score, inliersIndex] = helperComputeHomography(matchedPoints1, matchedPoints2)

[H, inliersLogicalIndex] = estgeotform2d( ...
    matchedPoints1, matchedPoints2, "projective", ...
    MaxNumTrials=2000, MaxDistance=8, Confidence=90);  
inlierPoints1 = matchedPoints1(inliersLogicalIndex);
inlierPoints2 = matchedPoints2(inliersLogicalIndex);
inliersIndex  = find(inliersLogicalIndex);
locations1 = inlierPoints1.Location;
locations2 = inlierPoints2.Location;
xy1In2     = transformPointsForward(H, locations1);
xy2In1     = transformPointsInverse(H, locations2);
error1in2  = sum((locations2 - xy1In2).^2, 2);
error2in1  = sum((locations1 - xy2In1).^2, 2);

outlierThreshold = 6;

score = sum(max(outlierThreshold - error1in2, 0)) + ...
        sum(max(outlierThreshold - error2in1, 0));
end

function [F, score, inliersIndex] = helperComputeFundamentalMatrix(matchedPoints1, matchedPoints2, intrinsics)

[E, inliersLogicalIndex] = estimateEssentialMatrix( ...
    matchedPoints1, matchedPoints2, intrinsics, ...
    MaxNumTrials=2000, MaxDistance=8, Confidence=90); 

F = intrinsics.K'\ E /intrinsics.K;
inlierPoints1 = matchedPoints1(inliersLogicalIndex);
inlierPoints2 = matchedPoints2(inliersLogicalIndex);
inliersIndex  = find(inliersLogicalIndex);

locations1 = inlierPoints1.Location;
locations2 = inlierPoints2.Location;
lineIn1   = epipolarLine(F', locations2);
error2in1 = (sum([locations1, ones(size(locations1, 1), 1)] .* lineIn1, 2)).^2 ./ ...
            sum(lineIn1(:, 1:2).^2, 2);
lineIn2   = epipolarLine(F, locations1);
error1in2 = (sum([locations2, ones(size(locations2, 1), 1)] .* lineIn2, 2)).^2 ./ ...
            sum(lineIn2(:, 1:2).^2, 2);

outlierThreshold = 4;

score = sum(max(outlierThreshold - error1in2, 0)) + ...
        sum(max(outlierThreshold - error2in1, 0));
end

function [isValid, xyzPoints, inlierIdx] = helperTriangulateTwoFrames(...
    pose1, pose2, matchedPoints1, matchedPoints2, intrinsics, minParallax)

camMatrix1 = cameraProjection(intrinsics, pose2extr(pose1));
camMatrix2 = cameraProjection(intrinsics, pose2extr(pose2));

[xyzPoints, reprojectionErrors, isInFront] = triangulate(matchedPoints1, ...
    matchedPoints2, camMatrix1, camMatrix2);
minReprojError = 1;
inlierIdx  = isInFront & reprojectionErrors < minReprojError;
xyzPoints  = xyzPoints(inlierIdx ,:);
ray1       = xyzPoints - pose1.Translation;
ray2       = xyzPoints - pose2.Translation;
cosAngle   = sum(ray1 .* ray2, 2) ./ (vecnorm(ray1, 2, 2) .* vecnorm(ray2, 2, 2));
isValid = all(cosAngle < cosd(minParallax) & cosAngle>0);
end
function mapPointSet = helperUpdateGlobalMap(...
    mapPointSet, vSetKeyFrames, vSetKeyFramesOptim)
posesOld     = vSetKeyFrames.Views.AbsolutePose;
posesNew     = vSetKeyFramesOptim.Views.AbsolutePose;
positionsOld = mapPointSet.WorldPoints;
positionsNew = positionsOld;
indices     = 1:mapPointSet.Count;
for i = indices
    majorViewIds = mapPointSet.RepresentativeViewId(i);
    poseNew = posesNew(majorViewIds).A;
    tform = affinetform3d(poseNew/posesOld(majorViewIds).A);
    positionsNew(i, :) = transformPointsForward(tform, positionsOld(i, :));
end
mapPointSet = updateWorldPoints(mapPointSet, indices, positionsNew);
end 
