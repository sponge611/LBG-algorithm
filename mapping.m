[codebook_map,text,rawdata] = xlsread("codebook.xlsx");
%disp(codebook_map(481,:))
%{
for i=1:512
    if codebook(i,:) ~= codebook_map(i,:)
        disp(i)
        disp("codebook: "+codebook(i,:));
        disp("codebook_map: "+codebook_map(i,:));
    end
end
%}
codebook_map = int32(codebook_map);

fin = fopen('892539.raw', 'r');
raw = fread(fin,128*128, 'uchar=>uint8');
fclose(fin);
lenna = reshape(raw,128,128);
lenna = lenna';
imwrite(lenna,"892539.jpg");


lenna_training_set = zeros(1024,16);
for i=1:1024
    start_pixel_row = floor((i-1)/32)*4 + 1;
    start_pixel_col = mod((i-1),32)*4 + 1;
    
    lenna_training_set(i,1:4) = lenna(start_pixel_row,start_pixel_col:start_pixel_col+3);
    lenna_training_set(i,5:8) = lenna(start_pixel_row+1,start_pixel_col:start_pixel_col+3);
    lenna_training_set(i,9:12) = lenna(start_pixel_row+2,start_pixel_col:start_pixel_col+3);
    lenna_training_set(i,13:16) = lenna(start_pixel_row+3,start_pixel_col:start_pixel_col+3);
end
%{
lenna_training_set = zeros(16384,16);
for i=1:16384
    start_pixel_row = floor((i-1)/128)*4 + 1;
    start_pixel_col = mod((i-1),128)*4 + 1;
    
    lenna_training_set(i,1:4) = lenna(start_pixel_row,start_pixel_col:start_pixel_col+3);
    lenna_training_set(i,5:8) = lenna(start_pixel_row+1,start_pixel_col:start_pixel_col+3);
    lenna_training_set(i,9:12) = lenna(start_pixel_row+2,start_pixel_col:start_pixel_col+3);
    lenna_training_set(i,13:16) = lenna(start_pixel_row+3,start_pixel_col:start_pixel_col+3);
end
%}
idx = zeros(1024,1);
for i=1:1024
    iteration_result = realmax;
    for j=1:512
    Kmeans = sum((codebook_map(j,:)-int32(lenna_training_set(i,:))).^2);
        if Kmeans < iteration_result
            idx(i,1) = j;
            iteration_result = Kmeans;
        end
    end
end
%{
idx = zeros(16384,1);
for i=1:16384
    iteration_result = realmax;
    for j=1:512
    Kmeans = sum((codebook_map(j,:)-int32(lenna_training_set(i,:))).^2);
        if Kmeans < iteration_result
            idx(i,1) = j;
            iteration_result = Kmeans;
        end
    end
end
%}
reconstruct_set = zeros(1024,16);
LBG_lenna = zeros(128,128);
for i=1:1024
    reconstruct_set(i,:) = codebook_map(idx(i,1),:);
end
%{
reconstruct_set = zeros(16384,16);
LBG_lenna = zeros(512,512);
for i=1:16384
    reconstruct_set(i,:) = codebook_map(idx(i,1),:);
end
%}
for i=1:1024
    start_pixel_row = floor((i-1)/32)*4 + 1;
    start_pixel_col = mod((i-1),32)*4 + 1;
    
    LBG_lenna(start_pixel_row,start_pixel_col:start_pixel_col+3) = reconstruct_set(i,1:4);
    LBG_lenna(start_pixel_row+1,start_pixel_col:start_pixel_col+3) = reconstruct_set(i,5:8);
    LBG_lenna(start_pixel_row+2,start_pixel_col:start_pixel_col+3) = reconstruct_set(i,9:12);
    LBG_lenna(start_pixel_row+3,start_pixel_col:start_pixel_col+3) = reconstruct_set(i,13:16);
end
%{
for i=1:16384
    start_pixel_row = floor((i-1)/128)*4 + 1;
    start_pixel_col = mod((i-1),128)*4 + 1;
    
    LBG_lenna(start_pixel_row,start_pixel_col:start_pixel_col+3) = reconstruct_set(i,1:4);
    LBG_lenna(start_pixel_row+1,start_pixel_col:start_pixel_col+3) = reconstruct_set(i,5:8);
    LBG_lenna(start_pixel_row+2,start_pixel_col:start_pixel_col+3) = reconstruct_set(i,9:12);
    LBG_lenna(start_pixel_row+3,start_pixel_col:start_pixel_col+3) = reconstruct_set(i,13:16);
end
%}
imwrite(uint8(LBG_lenna),'LBG_892539.jpg');
disp(psnr(uint8(LBG_lenna),lenna));