clear all;
clc;
close all;

im = imread('images/camMan.jpg');


if size(im,3) >= 3
    im = rgb2gray(im);
end

sg = 2;
k = 0.04;
N = 3;
np = 50;

[R] = harris(im, sg, k);
th = 0.10*max(R(:));
w  = 3; 

figure;imshow(R,[]);
title('R response');
hold;

[NMS] = nms_features(R,th,w);

[P,c] = corners (NMS, np);


figure;imshow(NMS,[]);
title('NMS');
hold;


show_corners(im,mat2gray(R),P,c);

%%
% Non - Maximum Suppression
% ------------------
% This function implements the non - maximum suppression algorithm
% for feature detection . All the non - maximum values inside a
% neighborhood is suppressed ((2 w + 1) x (2 w + 1) ) , remaining only
% the max local values . Avoid image border with width w . All values
% less than the limiar th are removed .
%
% Input
% ------------------
% M - Input matrix
% th - Threshold value
% w - Window size (2 w + 1) x (2 w + 1)
%
% Output
% ------------------
% NMS - Matrix with the non-maximum values suppressed
%%
function [NMS] = nms_features (M ,th ,w)
    [r, c] = size(M);
    
    NMS = zeros(r, c);
    
    a = w+1;
    for i=a:1:r-a
        for j=a:1:c-a
            NMS(i,j) = M(i,j);
            for x=i-w:i+w
                for z=j-w:j+w
                    if(M(i,j)<M(x,z) || M(i,j)<th)
                        NMS(i,j) = 0;
                    end
                end
            end
            
        end
    end
    
end

%%
% Corner Detection
% -------------------
% This function detects and returns the biggest np Harris ’ s Corners .
%
% Input
% -------------------
% R - Harris ’ s Corner Response
% np- max number of points to detect
%
% Output
% ------------------
% P- position of corners ( row , col ) . This is a matrix np x 2 ,
%    where np is the number of corners .
% c- Harris response of each corner . This is a ( np x 1) vector .2
%%
function[P, c] = corners (R, np)
    
    P = zeros(np,2);
    c = zeros(np,1);
    
    B = sort(R(:),'descend');
    i = 1;
    while i <=np
        val = B(i,1);
        if val == 0 || i == np + 1
            break
        end
        [rw,col] = find(R == val);
        for k=1:size(rw,1)
            P(i,1) = rw(k,1);
            P(i,2) = col(k,1);
            c(i,1) = val;
            i = i + 1;
            if i == np + 1
                break
            end
        end
    end
end


%%
% Harris’s Corner Response
% -------------------
% This function computes and returns the Harris’s Corner Response of an image .
%
% Input
% -------------------
% I- Grayscale image ( a matrix )
% sigma - Standard deviation of gaussian bluring
% k- k factor
%
% Output
% ------------------
% R- Harris response of each corner .
%%
function [R] = harris (I , sigma , k )
    [r, c] = size(I);
    %convertendo imagem para double
    imd = im2double(I);
    %definindo filtro gaussiano
    H = fspecial('gaussian',[3 3], sigma);
    %computando as derivadas da imagem
    [Ix, Iy] = imDerivate(imd);
    
    
    Ix2 = imfilter(Ix.^2, H);
    Iy2 = imfilter(Iy.^2, H);
    IxIy = imfilter(Ix.*Iy, H);
   
    
    figure,imshow(IxIy,[]);
    title('IxIy');
    figure,imshow(Ix2,[]);
    title('Ix2');
    figure,imshow(Iy2,[]);
    title('Iy2');
    
    
    R = zeros(r, c);
    
    %computando a matriz M para cada pixel em relação aos seus vizinhos
    for i=2:1:r-1
        for j=2:1:c-1
            Ix2_  = sum(sum(Ix2(i-1:i+1,j-1:j+1)));
            Iy2_  = sum(sum(Iy2(i-1:i+1,j-1:j+1)));
            IxIy_ = sum(sum(IxIy(i-1:i+1,j-1:j+1)));
            
            M = [Ix2_ IxIy_;IxIy_ Iy2_];
            
            R(i,j) = det(M) - k*trace(M).^2; 
        end
    end
    
        
end

%%
% imgradient
% Input :
% I - input image
% Output :
% Gx - Output image of x gradient
% Gy - Output image of y gradient
%%
function[Gx, Gy] = imDerivate(I)
    p = [0.037659 0.249153 0.426375 0.249153 0.037659];
    d = [0.109604 0.276691 0.00000 -0.276691 -0.109604];
    
    Gx = zeros(size(I));
    Gy = zeros(size(I));
    
    for i = 1:size(I,1)
        Gx(i,:) = conv(I(i,:),d,'same');
        Gy(i,:) = conv(I(i,:),p,'same');
    end
    
    for i = 1:size(I,2)
        Gx(:,i) = conv(Gx(:,i),p,'same');
        Gy(:,i) = conv(Gy(:,i),d,'same');
    end
end

%%
% Show Corners
% -------------------
%
% Input
% -------------------
% I - input image ( a matrix )
% P - position of corners ( row , col ) . This is a matrix Kx2 ,
%     where K is the number of corners .
% c - Harris response of each corner . This is a Kx1 vector .
%
% Output
% ------------------
%%
function [] = show_corners (I, R, P , c)

    m = min(c(:));
    M = max(c(:));
    
    for i=1:size(P,1)
        d = abs((20*c(i,1))/M);
        
        if(d<=3)
            x1 = P(i,2) - d/2;
            y1 = P(i,1) - d/2;
            I = insertShape(I,'Rectangle',[x1 y1 3 3],'Color','red');
            R = insertShape(R,'Rectangle',[x1 y1 3 3],'Color','red');
        else
            x1 = P(i,2) - d/2;
            y1 = P(i,1) - d/2;
            I = insertShape(I,'Rectangle',[x1 y1 d d],'Color','red');
            R = insertShape(R,'Rectangle',[x1 y1 d d],'Color','red');
        end
        
    end
    
    figure;imshow(I,[]);
    title('Original image');
    hold;
    figure;imshow(R,[]);
    title('Harris response');
    hold;
    
end