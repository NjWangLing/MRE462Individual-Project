function [isDetected, loopKeyFrameIds] = helperCheckLoopClosure(vSetKeyFrames, ...
    currKeyframeId, loopDatabase, currFeatures, loopEdgeNumMatches)
%helperCheckLoopClosure detect loop candidates key frames by retrieving
%   visually similar images from the feature database.
%
%   This is an example helper function that is subject to change or removal 
%   in future releases.

%   Copyright 2019-2024 The MathWorks, Inc.
%#codegen

% Retrieve all the visually similar key frames
[candidateViewIds, similarityscores] = detectLoop(loopDatabase, currFeatures);
candidateViewIds = double(candidateViewIds');
similarityscores = double(similarityscores');

% Compute similarity between the current key frame and its strongly-connected 
% key frames. The minimum similarity score is used as a baseline to find 
% loop candidate key frames, which are visually similar to but not connected 
% to the current key frame
covisViews          = connectedViews(vSetKeyFrames, currKeyframeId);
covisViewsIds       = covisViews.ViewId;
strongCovisViews    = connectedViews(vSetKeyFrames, currKeyframeId, 'MinNumMatches', loopEdgeNumMatches);
strongCovisViewIds  = strongCovisViews.ViewId;

% Retrieve the top 10 similar connected key frames
[~,~,ib] = intersect(strongCovisViewIds, candidateViewIds);
minScore     = min(similarityscores(ib));

[loopKeyFrameIds,ia] = setdiff(candidateViewIds, covisViewsIds, 'stable');

% Scores of non-connected key frames
candidateScores  = similarityscores(ia); % Descending

if ~isempty(ia) && ~isempty(ib)
    bestScore       = candidateScores(1);
    
    % Score must be higher than the 75% of the best score 
    isValid         = candidateScores > max(bestScore*0.75, minScore);
    
    loopKeyFrameIds = loopKeyFrameIds(isValid);
else
    loopKeyFrameIds = zeros(coder.ignoreConst(0), coder.ignoreConst(0), class(loopKeyFrameIds));
end

% Loop candidates need to be consecutively detected
minNumCandidates = 3; % At least 3 candidates are found
if size(loopKeyFrameIds,1) >= minNumCandidates
    groups = nchoosek(loopKeyFrameIds, minNumCandidates);
    consecutiveGroups = groups(max(groups,[],2) - min(groups,[],2) < 4, :);
    if ~isempty(consecutiveGroups) % Consecutive candidates are found
        loopKeyFrameIds = consecutiveGroups(1,:);
        isDetected = true;
    else
        isDetected = false;
    end
else
    isDetected = false;
end
end