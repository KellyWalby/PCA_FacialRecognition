%Philip Varkey, Kelly Walby, Austin Johns, Kyle KirkPatrick, and Duyen Vu
%Facial Recognition
clc
clear

%% values of the number of images in the folder, the average mean of the images
Set = 20;                                                                   %number of images
u_mean = 120;                                                               %average mean of the images
u_stnrd_dev = 60;                                                           %average standard deveation of the images

%Sets the folder with which the database is taken and reads the image in the folder 
folder = cd;                                                                %current folder
newfolder =(fullfile(cd,'Training_Database'));                              %new folder where you want to go
cd(newfolder)                                                               %Goes into the folder 
names = dir;
names = names(1:end);                                                       %Recieves on all names in the folder depending on where the folder starts
hh = length(names); 
mkdir training_Set                                                          %Creates a new file called training_Set 

%initializes empty matrices
S = [];
u = [];

%% Reads in all images and alters them into grayscale
for i = 3:Set+2
    
    truvecs(i).names = names(i).name;                                       %the new vector that got the fields
    name = truvecs(i).names;                                                %returns to the current folder
    img = imread(name);                                                     %Read the image
    eval('img');                                                            %Evaluates each image by reading the img using imread and eval
    S = img;                                                                %Sets matrix S to img
    imshow(S)                                                               %Show the image of S using imshow
    
    gray = rgb2gray(S);                                                     %Removes the color of the image using rgb2gray
    BW = gray;                                                              %Sets matrix BW to gray

    %% minimizies the background 
    [n1, n2] = size(BW);
    r = floor(n1/10);                                                       %rounds the dimension of x/10 down using floor
    c = floor(n2/10);                                                       %rounds the dimension of y/10 down using floor
    x1 = 1;
    x2 = r;
    s = r*c;
    for ib = 1:10
        y1 = 1;
        y2 = c;                                                             %sets the spacing of the image 
        for j = 1:10
            if (y2 <= c || y2 >= 9*c) || (x1 == 1 || x2 == r*10)
                loc = find(BW(x1:x2, y1:y2)== 0);                           %Find the inividual columns of x1,x2,y1,y2
                [o, p] = size(loc);                                         %Sets the matrix containing o and p to the size of loc
                pr = o*100/s;
                if pr <= 100
                    BW(x1:x2, y1:y2) = 0;                                   %Sets the given area to zero so as to remove its color
                    r1 = x1;
                    r2 = x2;
                    s1 = y1;
                    s2 = y2;
                    pr1 = 0;
                end
            end
                y1 = y1 + c;
                y2 = y2 + c;
        end
    x1 = x1 + r;                                                            %moves around the image by increasing the values of x1 and x2
    x2 = x2 + r;
    end
    
    %% the face is detected and the image is resized to the rectangle box around the face
    face = bwlabel(BW,8);                                                   %creates a label using bwlabel
    BB = regionprops(face, 'BoundingBox');                                  %Sets the area around the face
    BB1 = struct2cell(BB);                                                  %converts the label to a cell array using struct2cell
    BB2 = cell2mat(BB1);                                                    %converts the cell arrray to one large array using cell2mat
    [s1, s2] = size(BB2);                                                   %Creates a matrix that is the size of the array
    mx = 0;
    for k = 3:4:s2-1                                                        %for loop to iterate over s2
        p = BB2(1,k)*BB2(1,k+1);
        if p>mx && (BB2(1,k)/BB2(1,k+1))<1.8
            mx = p;
            j = k;
        end
    end
    rectangle('Position',[BB2(1,j-2),BB2(1,j-1),BB2(1,j),BB2(1,j+1)],'EdgeColor','r' ) %Creates a rectangular box to be around the person's face
    
    BW = imcrop(BW,[BB2(1,j-2),BB2(1,j-1),BB2(1,j),BB2(1,j+1)]);            %Crops the image to the rectangular box using imcrop
    
    %% columns are removed for any remaining background
    for ux = 1:r                                                            %for loop to iterate over the value of r
        BW(ux,:) =[];                                                       %sets the rows in r to zero
        BW(:,ux) = [];                                                      %Sets the columns in r to zero
    end
    
    for ir = 1:size(BW)                                                     %for loop to iterate over the size of BW
        if BW(ir) < 0                                                       %if statement to see if the values of BW are less than zero
            BW(ir) = -BW(ir);                                               %Sets the negative values to their positive counterparts
        end
    end
    img = BW;                                                               %Sets the orginial image to BW
    img = imresize(BW,[155 155]);                                           %Resizes the image to 155x155 using imresize
    
    %% adjusts the image to only show predominant features
    imshow(img)                                                             %Shows the cropped image using imshow
    gray = img;                                                             %Sets matrix gray to img
    background = gray < 0.1;                                                %backgournd is set if the pixel is less than a certain number
    gray(background) = 255;                                                 %Set background to 255 (white)
   
    S = gray;                                                               %Sets matrix S to gray
    [x,y] = size(S);                                                        %creates a matrix with two vectors each with the lave of the length of S
    whiteground = S > 230;                                                  %Sets white ground to the values in S that are greater than 230
    S(whiteground) = 0;                                                     %Set the values equal to whiteground to zero
    background = S > 125;                                                   %backgournd is set if the pixel is less than a certain number
    S(background) = 255;                                                    %Set background to 255 (white)
    imshow(S)                                                               %Show the image again with any alteration
    
    %% calculates the mean of each image
    for k = 1:size(S,2)                                                     %for loop to iterate over the size of S
        temp = double(S(k,:));                                              %Creates a double version of S to iterate over
        a = mean(temp);                                                     %Calculates the mean of the column using mean
        dev = std(temp);                                                    %Calculates the standard deviation of temp using std
        W(k,:) = sqrt((temp-a)*u_stnrd_dev/(dev+u_mean));                   %Square roots the value and stores it in a new column
        W(k,:) = real(W(k,:));                                              %converts the column to a real value column only using real
    end
    imshow(W)
    
    %covariance matrix, L = transpose(A)*A
    L = transpose(W)*W;                                                     %creates the convariance matrix by multiplying its tranpose by itself

    %% Creates the eigen values of the covariance matrix
    %remove the m - n eigen values and then get a square matrix of U and V 
    %Gets the rank of the matrix
    rank = sprank(double(L));                                               %Sets rank to the structural rank of L
    vec = [];                                                               %Creates an empty matrix vec
    val = [];                                                               %Creates an empty matrix val
    for j = 1:Set                                                           %for loop to iterate over Set
        [vec, val] = eigs(L);                                               %Takes the eigen value of the new matrix using eig
        
        %if a value of e is less than the rank, it is replaced with zero
        e = sort(real(vec),'descend');                                      %Sorts the eigen values from greatest to least using sort and descend
        for l = 1:size(e)                                                   %for loop to iterate over the size of e
            if (e(l,:) > rank)                                              %if statement to see if the position of the kth column of e is greater than the rank
                e(l,:) = 0;                                                 %set the value to zero
            end
        end
        vec = e;                                                            %Sets vector vec to e
    end

    %% Creates the eigen vectors and normalizes them
    %Eigenvectors of L
    
    truvec = [];                                                            %Creates an empty matrix truvec
    truval = [];                                                            %Creates an empty matrix truval
    
    for tru = 1:size(vec,2)                                                 %for loop to iterate over the size of vec
        if(val(tru,tru) > rank)                                             %if statement to check if the individual value of val is less than rank
            truvec = [truvec vec(:,tru)];                                   %adds the values of vec to truvec
            truval = [truval val(tru,tru)];                                 %adds the values of val to truval
        end
    end
    
    [B, index] = sort(truval, 'descend');                                   %Sets B and index to the value of truval descending 
    ind = zeros(size(B));                                                   %Creates a matrix the size of index filled with zeros 
    
    dtemp = zeros(size(B));                                                 %Creates a matrix the size of index filled with zeros
    vtemp = zeros(size(truvec));                                            %Creates a matrix the size of truvec filled with zeros
    len = length(index);                                                    %len is set to the length of index
    
    for gth=1:len                                                           %for loop to iterate over the length of index
       dtemp(gth) = B(gth);                                                 %set dtemp to the len+1-gth value of B
       vtemp(:,gth) = truvec(:,gth);                                        %Sets the ind(gth)th of vtemp to the gth column of truvec
    end
    
    truvec = sort(dtemp, 'descend');                                        %truvec is set to the matrix dtemp
    truval = sort(vtemp, 'descend');                                        %truval is set to the matrix vtemp
    
    %Normalization of eigenvectors
    for nor = 1:size(truvec,2)                                              %for loop to iterate over the size of truvec
        kk = truval(:,nor);                                                 %sets kk to the nor'th columns of truval
        temp1 = sqrt(sum(kk.^2));                                           %sets temp to the sum of kk squared
        truval(:,nor) = truval(:,nor)./temp1;                               %Set the nor'th value of truval to itself divided by temp
    end
    
    %Eigenvectors of covariance matrix
    for eig = 1:size(truval,2)                                              %for loop to iterate over the size of truvec
        temp2 = sqrt(truval(eig));                                          %sets temp to the square root of the eig'th value of truval
        u = [u (L*truval(:,eig))./temp2];                                   %add (W times the eig'th column of truvalue divided by temp) to u
    end
    u = sort(abs(u(:,1)),'descend');                                        %u is set to the absolute value of the first column of u
       
    %Normalization of eigenvectors
    for eivec = 1:size(u,2)                                                 %for loop to iterate over the size of u
        kk = u(:,eivec);                                                    %kk is set to the eivec'th column of u
        temp3 = sqrt(sum(kk.^2));                                           %temp is set to the square rook of the sum of kk squared
        u(:,eivec) = u(:,eivec)./temp3;                                     %u is set to the eivec'th columns of u is divided by temp
    end
    
    for unit = 1:size(u,2)                                                  %for loop to iterate through the vector u
        if (u(unit) < .001)                                                 %if statement to check if the unit'th value in u is less than 0.001
            u(unit) = 0;                                                    %Sets the value to zero
        end
    end
   
    %% Finds the weight of each face in the training set.
    omega = [];                                                             %Creates an empty matrix called omega
    for weight=1:size(W,2)                                                  %for loop to iterate over the size of W
        WW=[];                                                              %Creates an empty matrix called WW    
        for weight_index=1:size(u,2)                                        %for loop to iterate over the size of u
            t = double(u(:,weight_index)');                                 %Converts the value of u to a double    
            WeightOfImage = dot(t,W(:,weight)');                            %Sets WeightOfImage to the dot product of t and W
            WW = [WW; WeightOfImage];                                       %WeightOfImage is added to the matrix WW
        end
        omega = [omega WW];                                                 %WW is added to the matrix omega
    end
    omega = sort(omega, 'descend');                                         %sorts omega in a descending order using sort
    
    final_name = name;                                                      %Sets final_name to the nmae of the image
    filename = sprintf('%s.csv', final_name);                               %creates a csv file using sprintf
    csvwrite(filename, omega');                                             %Saves omega witin the csv file using filewrite
    movefile(filename, 'C:\Users\PhilV\Desktop\College\2019 Fall Semester\Linear Algebra\Training_Database\training_set');
end
%% New database for testing
movefile('training_set', 'C:\Users\PhilV\Desktop\College\2019 Fall Semester\Linear Algebra\Final_Database')
%% finds a new folder and compare 
new_set = 120;
cd 'C:\Users\PhilV\Desktop\College\2019 Fall Semester\Linear Algebra\'      %moves to a new folder
newfolder = cd;
newfolder =(fullfile(cd,'Final_Database'));                                 %new folder where you want to go
cd(newfolder)                                                               %enters the new folder
New_names = dir;
New_names  = New_names(1:end);                                              %Recieves on all names in the folder depending on where the folder starts
new__hh = length(New_names);                                                %Sets new_hh to the length of New_names
New_S = [];                                                                 %Initializes New_S as a empty matrix 
New_u = [];                                                                 %Initializes New_u as a empty matrix

for i = 3:new_set+2
    
    New_truvecs(i).New_names = New_names(i).name;                           %the new vector that got the fields
    New_name = New_truvecs(i).New_names;                                    %returns to the current folder
    New_img = imread(New_name);                                             %Read the image
    eval('New_img');                                                        %Evaluates each image by reading the img using imread and eval
    New_S = New_img;                                                        %Sets New_s to New_img
    K = New_img;                                                            %Sets a new matrix K to New_img
    figure                                                                  %Creates a new figure
    imshow(New_S)                                                           %shows New_S usinf imshow
    
    New_gray = rgb2gray(New_S);                                             %Sets New_gray to New_S grayscaled using rgb2gray
    FBW = New_gray;                                                         %Sets FBW to New_gray

    %% minimizing the background 
    [n1, n2] = size(FBW);
    r = floor(n1/10);                                                       %rounds the dimension of x/10 down using floor
    c = floor(n2/10);                                                       %rounds the dimension of y/10 down using floor
    x1 = 1;
    x2 = r;
    s = r*c;
    for New_i = 1:10
        y1 = 1;
        y2 = c;                                                             %sets the spacing of the 
        for New_j = 1:10
            if (y2 <= c || y2 >= 9*c) || (x1 == 1 || x2 == r*10)
                loc = find(FBW(x1:x2, y1:y2)== 0);                          %Find the inividual columns of x1,x2,y1,y2
                [o, p] = size(loc);                                         %Sets the matrix containing o and p to the size of loc
                pr = o*100/s;
                if pr <= 100
                    FBW(x1:x2, y1:y2) = 0;
                    r1 = x1;
                    r2 = x2;
                    s1 = y1;
                    s2 = y2;
                    pr1 = 0;
                end
            end
                y1 = y1 + c;
                y2 = y2 + c;
        end
    x1 = x1 + r;
    x2 = x2 + r;
    end
    
    %% the face is detected and the image is resized to the rectangle box around the face
    face = bwlabel(FBW,8);
    new_BB = regionprops(face, 'BoundingBox');                              %Creates a box around the face using regionprops
    new_BB1 = struct2cell(new_BB);                                          %converts new_BB1 to a cell array using struct2cell 
    new_BB2 = cell2mat(new_BB1);                                            %converts new_BB1 toa single matrix using cell2mat
    [new_s1, new_s2] = size(new_BB2);                                       %sets the dimension of new_BB2 to new_s1 and new_s2
    mx = 0;
    for k = 3:4:new_s2-1                                                    %for loop to iterate over new_s2
        p = new_BB2(1,k)*new_BB2(1,k+1);                                    %multiplies the kth value of new_BB2 to the k+1 value of BB_2
        if p>mx && (new_BB2(1,k)/new_BB2(1,k+1))<1.8                        %if statement to check if p > mx and the kth value New_BB2 divided by the k+1th value of New_BB2 is less than 1.8
            mx = p;
            j = k;
        end
    end
    rectangle('Position',[new_BB2(1,j-2),new_BB2(1,j-1),new_BB2(1,j),new_BB2(1,j+1)],'EdgeColor','r' )
    
    FBW = imcrop(FBW,[new_BB2(1,j-2),new_BB2(1,j-1),new_BB2(1,j),new_BB2(1,j+1)]); %Crops FBW to the size of the of the rectangular box using imcrop
    
    %% columns are removed for any remaining background
    for ux = 1:r                                                            %for loop to iterate over the current value of r
        FBW(ux,:) =[];                                                      %removes the uxth row of FBW
        FBW(:,ux) = [];                                                     %removes the uxth columns of FBW
    end
    
    for ir = 1:size(FBW)                                                    %for loop to iterate over the size the
        if FBW(ir) < 0                                                      %if statement to see if the value of FBW is negative
            FBW(ir) = -FBW(ir);                                             %Set the negative value to its positive counter part 
        end
    end
    New_img = imresize(FBW,[155 155]);                                      %sets New_img to the values of FBW resized to a 155x155 matrix using imresize
    imshow(New_img)                                                         %Shows the new image using imshow
    
    %% adjusts the pixels to only show predominant feature
    imshow(New_img)                                                         %Shows New_img as a image using imshow
    New_gray = New_img;                                                     %Sets matrix New_gray to New_img
    background = New_gray < 0.1;                                            %backgournd is set if the pixel is less than a certain number
    New_gray(background) = 255;                                             %Set background to 255 (white)
   
    New_S = New_gray;                                                       %Sets matrix New_S to New_gray
    [x,y] = size(New_S);                                                    %creates a matrix with two vectors each with the lave of the length of S
    whiteground = New_S > 230;                                              %Sets whiteground to any values in New_S that are greater than 230
    New_S(whiteground) = 0;                                                 %Set each value in New_S that is equal to whiteground to zero
    background = New_S > 125;                                               %backgournd is set if the pixel is less than a certain number
    New_S(background) = 255;                                                %Set background to 255 (white)
    imshow(New_S)                                                           %Show the image again with any alteration

    %% calculates the mean of each image
    for k = 1:size(New_S,2)                                                 %for loop to iterate over the size of New_S
        newtemp = double(New_S(k,:));                                       %Creates a double version of S to iterate over
        a = mean(newtemp);                                                  %Calculates the mean of the column using mean
        dev = std(newtemp);                                                 %Sets dev to the standard deviation of newtemp
        New_W(k,:) = sqrt((newtemp-a)*u_stnrd_dev/(dev+u_mean));            %Square roots the value and stores it in a new column
        New_W(k,:) = real(New_W(k,:));                                      %converts the column to a real value column only using real
    end
    imshow(New_W)                                                           %Shows the image of New_W using imshow
    %covariance matrix, L = transpose(A)*A
    New_L = transpose(New_W)*New_W;

    %% Creates the eigen values of the covariance matrix and evaluates the feature vectors
    %remove the m - n eigen values and then get a square matrix of U and V 
    
    %Gets the rank of the matrix
    New_rank = sprank(double(New_L));                                       %The structural rank of the covariance matrix is found using sprank
    New_vec = [];                                                           %Creates an empty matrix New_vec 
    New_val = [];                                                           %Creates a empty matrix New_val
    for wj = 1:new_set                                                      %for loop to iterate over new_set
        [New_vec, New_val] = eigs(New_L);                                   %Takes the eigen value of the new matrix using eig
       
        %if a value of e is less than the rank, it is replaced with zero
        New_e = sort(real(New_vec),'descend');                              %Sorts the eigen values from greatest to least using sort and descend
        for wl = 1:size(New_e)                                              %iterates through e
            if (New_e(wl,:) > New_rank)                                     %finds any values less than 9.0600e-5
                New_e(wl,:) = 0;                                            %set the value to zero
            end
        end
        New_vec = New_e;                                                    %New_vec is set to the value of New_e
    end

    %% Creates the eigen vectors and normalizes them
    %Eigenvectors of L
    
    New_truvec = [];                                                        %Creates an empty matrix truvec
    New_truval = [];                                                        %Creates an empty matrix truval
    
    for New_tru = 1:size(New_vec,2)                                         %for loop to iterate over the size of vec
        if(New_val(tru,tru) > rank)                                         %if statement to check if the individual value of val is less than rank
            New_truvec = [New_truvec vec(:,New_tru)];                       %adds the values of vec to truvec
            New_truval = [New_truval val(New_tru,New_tru)];                 %adds the values of val to truval
        end
    end
    
    [New_B, New_index] = sort(New_truval, 'descend');                       %Sets B and index to the value of truval descending 
    New_ind = zeros(size(B));                                               %Creates a matrix the size of index filled with zeros 
    
    New_dtemp = zeros(size(B));                                             %Creates a matrix the size of index filled with zeros
    New_vtemp = zeros(size(New_truvec));                                    %Creates a matrix the size of truvec filled with zeros
    New_len = length(New_index);                                            %len is set to the length of index
    
    for New_gth=1:New_len                                                   %for loop to iterate over the length of index
       New_dtemp(New_gth) = New_B(New_gth);                                 %set dtemp to the len+1-gth value of B
       New_vtemp(:,New_gth) = New_truvec(:,New_gth);                        %Sets the ind(gth)th of vtemp to the gth column of truvec
    end
    New_truvec = sort(New_dtemp,'descend');                                 %truvec is set to the matrix dtemp
    New_truval = sort(New_vtemp,'descend');                                 %truval is set to the matrix vtemp
    
    %Normalization of eigenvectors
    for n_nor = 1:size(New_truvec,2)                                        %for loop to iterate over the size of truvec
      New_kk = New_truval(:,n_nor);                                         %sets kk to the nor'th columns of truval
      New_temp1 = sqrt(sum(New_kk.^2));                                     %sets temp to the sum of kk squared
      New_truval(:,n_nor) = New_truval(:,n_nor)./New_temp1;                 %Set the nor'th value of truval to itself divided by temp
    end
    
    %Eigenvectors of C matrix
    for eig = 1:size(New_truvec,2)                                          %for loop to iterate over the size of truvec
        New_temp2 = sqrt(New_truval(eig));                                  %sets temp to the square root of the eig'th value of truval
        New_u = [New_u (New_L*New_truval(:,eig))./New_temp2];                                    %add (W times the eig'th column of truvalue divided by temp) to u
    end
    New_u = sort(abs(New_u(:,1)),'descend');                                %u is set to the absolute value of the first column of u
    
    %Normalization of eigenvectors
    for eivec = 1:size(New_u,2)                                             %for loop to iterate over the size of u
       New_kk = New_u(:,eivec);                                             %kk is set to the eivec'th column of u
       New_temp3 = sqrt(sum(New_kk.^2));                                    %temp is set to the square rook of the sum of kk squared
       New_u(:,eivec) = New_u(:,eivec)./New_temp3;                          %u is set to the eivec'th columns of u is divided by temp
    end
    
    for new_unit = 1:size(New_u,2)                                          %for loop to iterate over the size of New_u
        if (New_u(new_unit) < .001)                                         %if statement to check if the value of New_u is less than 0.001
            New_u(new_unit) = 0;                                            %Sets the nuw_unit value of New_u to zero
        end
    end
    
    %% Finds the weight of each face in the database.
    new_omega = [];                                                         %initalizes the matrix new_omega
    for new_weight=1:size(New_W,2)                                          %for loop to iterate over the length of New_W
        New_WW=[];                                                          %initalizes new matrix New_WW    
        
        for new_weight_index=1:size(New_u,2)                                %for loop to iterate over the size of New_u
            t = double(New_u(:,new_weight_index)');                         %Sets t to the double of New_u the new_weight_indexth column transposed
            New_WeightOfImage = dot(t,New_W(:,new_weight)');                %Sets New_WeightOfImage to the dot product of t and New_W
            New_WW = [New_WW; New_WeightOfImage];                           %Adds the new value to New_WW
        end
        
        new_omega = [new_omega New_WW];                                     %Adds New_WW to new_omega
    end
    new_omega = sort(new_omega, 'descend');                                 %Sorts new_omega in a desending order using sort
    
    %% Find Euclidean distance
    eu = [];                                                                %Initializes an empty matrix eu
    thenames = {};                                                          %Initializes a new set called thenames 
    myFolder = 'C:\Users\PhilV\Desktop\College\2019 Fall Semester\Linear Algebra\Final_Database\training_Set';
    filePattern = fullfile(myFolder, '*.csv');                              %Finds all files that have the .csv end 
    theFiles = dir(filePattern);                                              
    
    for k = 1 : size(theFiles,1)                                            %for loop to iterate over the size of theFiles
        baseFileName = theFiles(k).name;                                    %Set baseFileName to the kth name of theFiles
        fullFileName = fullfile(myFolder, baseFileName);                    %Sets fullFileName to the values within the file
        thenames(k) = textscan(baseFileName,'%s','Delimiter',' ')';         %Sets the kth value of j to the name of the file
        thelist = xlsread(fullFileName);                                    %Sets thelist to the values in fullFileName using xlsread
        
        for euclid = 1:size(thelist,2)                                      %for loop to iterate over the size of thelist
            thelist = double(thelist);                                      %sets thelist to the doulbe of thelist
            thelist = thelist';                                             %Sets thelist to thelist transposed
            DiffWeight = new_omega-thelist;                                 %Sets DiffWeight to the difference of new_omega and thelist
            mag = norm(DiffWeight);                                         %mag is set to the Normalized DiffWeight 
            eu = [eu mag];                                                  %Adds the value of mag to the matrix eu
        end 
    end
    
    %% shows the closest image to the testing image
    MinimumValue = min(eu);                                                 %Sets MinimumValue to the min of eu 
    position = ismember(eu,MinimumValue);                                   %Sets position to the position MinimumValue in eu
    eu = strcat(thenames, num2str(eu));                                     %Creates a logical vector 
    final = eu(position);                                                   %final is set to the position value of eu
    
    figure                                                                  %creates a new figure
    subplot(1,2,1)                                                          %creates a subplot of the figure
    imshow(K)                                                               %shows the orginial image using imshow
    
    [final] = final{1:1,1:1};                                               %Takes the the cell value of final and sets final to the value
    [final_txt] = final{1,1};                                               %Takes the {1,1} value of final and sets it to final_txt
    final_txt = final_txt(1:end-4);                                         %Removes the .csv of the name of the image
    end_img = imread(final_txt);                                            %Sets end_img to the imread of final_txt 
    
    subplot(1,2,2)                                                          %Creates a subplot of the figure
    imshow(end_img);                                                        %Displays end_img using imshow
    title('the closest image')                                              %Adds a title to the figure
end