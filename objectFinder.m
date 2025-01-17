%% Alexander Bennett w21005417 MATLAB Code
%select tbe image file of the object you wish to detect, and an image of
%the environment the object is in
objectInSceneColour = imread('scene 2.jpeg');
objectFindColour = imread('tonychocolate.jpeg');

%Convert the selected images into greyscale to allow for compatability with
%functions
objectFind = rgb2gray(objectFindColour);
objectInScene = rgb2gray(objectInSceneColour);

%Show the user the object they wish to detect
figure;
imshow(objectFindColour);
title('Object to find');

%Detect feature points of the selected images through use of CNN
objectFindPoints = detectSURFFeatures(objectFind);
objectInScenePoints = detectSURFFeatures(objectInScene);

%Select and show 200 of the strongest features of the object
figure; 
imshow(objectFindColour);
title('Features');
hold on;
plot(selectStrongest(objectFindPoints, 200));

%Select and show 300 of the strongest features of the entire scene
figure; 
imshow(objectInSceneColour);
title('Feature Points');
hold on;
plot(selectStrongest(objectInScenePoints, 350));

% Select information on the feature points from both images
[obFeatures, obPoints] = extractFeatures(objectFind, objectFindPoints);
[picFeatures, picPoints] = extractFeatures(objectInScene, objectInScenePoints);

% Find matching features based on the information extracted previously
obMatch = matchFeatures(obFeatures, picFeatures);

% Display strong matches
matchedObPoints = obPoints(obMatch(:, 1), :);
matchedPicPoints = picPoints(obMatch(:, 2), :);
figure;
showMatchedFeatures(objectFind, objectInScene, matchedObPoints, matchedPicPoints, 'montage');
title('Points that were found to match');

%Based on these matches, find the object in the scene
%"estgeotform2d" elimates outliers and finds the transformation between the
%sample image of the object and the object in the scene
[tform, inlierIdx] = estgeotform2d(matchedObPoints, matchedPicPoints, 'affine');
corrObPoints   = matchedObPoints(inlierIdx, :);
corrPicPoints = matchedPicPoints(inlierIdx, :);

%Show the matches with no outliers
figure;
showMatchedFeatures(objectFind, objectInScene, corrObPoints, corrPicPoints, 'montage');
title('Matched Points without annomalies');

%the polygon variable below creates a four sided shape that surrounds the
%object the code is trying to find, ultimately highlighting it for the user
objectPolygon = [1, 1; size(objectFind, 2), 1; size(objectFind, 2), size(objectFind, 1); 1, size(objectFind, 1); 1, 1];       

newObPolygon = transformPointsForward(tform, objectPolygon);   

%finally we show these results to the user, allowing the user to see where
%the oject is in the image
figure;
imshow(objectInSceneColour);
hold on;
line(newObPolygon(:, 1), newObPolygon(:, 2), Color='y');
title('Detected Object');
