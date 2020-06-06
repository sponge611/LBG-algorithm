fin = fopen('lenna.raw', 'r');
raw = fread(fin,512*512, 'uchar=>uint8');
fclose(fin);
lenna = reshape(raw,512,512);
lenna = lenna';
imwrite(lenna,"lenna.jpg");

fin = fopen('mandrill.raw', 'r');
raw = fread(fin,512*512, 'uchar=>uint8');
fclose(fin);
mandrill = reshape(raw,512,512);
mandrill = mandrill';
imwrite(mandrill,"mandrill.jpg");

fin = fopen('scene.raw', 'r');
raw = fread(fin,512*512, 'uchar=>uint8');
fclose(fin);
scene = reshape(raw,512,512);
scene = scene';
imwrite(scene,"scene.jpg");

fin = fopen('tiffany.raw', 'r');
raw = fread(fin,512*512, 'uchar=>uint8');
fclose(fin);
tiffany = reshape(raw,512,512);
tiffany = tiffany';
imwrite(tiffany,"tiffany.jpg");

codebook = zeros(512,16);
lenna_training_set = zeros(16384,16);
mandrill_training_set = zeros(16384,16);
scene_training_set = zeros(16384,16);
tiffany_training_set = zeros(16384,16);
for i=1:16384
    start_pixel_row = floor((i-1)/128)*4 + 1;
    start_pixel_col = mod((i-1),128)*4 + 1;
    
    lenna_training_set(i,1:4) = lenna(start_pixel_row,start_pixel_col:start_pixel_col+3);
    lenna_training_set(i,5:8) = lenna(start_pixel_row+1,start_pixel_col:start_pixel_col+3);
    lenna_training_set(i,9:12) = lenna(start_pixel_row+2,start_pixel_col:start_pixel_col+3);
    lenna_training_set(i,13:16) = lenna(start_pixel_row+3,start_pixel_col:start_pixel_col+3);
end

for i=1:16384
    start_pixel_row = floor((i-1)/128)*4 + 1;
    start_pixel_col = mod((i-1),128)*4 + 1;
    
    mandrill_training_set(i,1:4) = mandrill(start_pixel_row,start_pixel_col:start_pixel_col+3);
    mandrill_training_set(i,5:8) = mandrill(start_pixel_row+1,start_pixel_col:start_pixel_col+3);
    mandrill_training_set(i,9:12) = mandrill(start_pixel_row+2,start_pixel_col:start_pixel_col+3);
    mandrill_training_set(i,13:16) = mandrill(start_pixel_row+3,start_pixel_col:start_pixel_col+3);
end

for i=1:16384
    start_pixel_row = floor((i-1)/128)*4 + 1;
    start_pixel_col = mod((i-1),128)*4 + 1;
    
    scene_training_set(i,1:4) = scene(start_pixel_row,start_pixel_col:start_pixel_col+3);
    scene_training_set(i,5:8) = scene(start_pixel_row+1,start_pixel_col:start_pixel_col+3);
    scene_training_set(i,9:12) = scene(start_pixel_row+2,start_pixel_col:start_pixel_col+3);
    scene_training_set(i,13:16) = scene(start_pixel_row+3,start_pixel_col:start_pixel_col+3);
end

for i=1:16384
    start_pixel_row = floor((i-1)/128)*4 + 1;
    start_pixel_col = mod((i-1),128)*4 + 1;
    
    tiffany_training_set(i,1:4) = tiffany(start_pixel_row,start_pixel_col:start_pixel_col+3);
    tiffany_training_set(i,5:8) = tiffany(start_pixel_row+1,start_pixel_col:start_pixel_col+3);
    tiffany_training_set(i,9:12) = tiffany(start_pixel_row+2,start_pixel_col:start_pixel_col+3);
    tiffany_training_set(i,13:16) = tiffany(start_pixel_row+3,start_pixel_col:start_pixel_col+3);
end


for i=1:512
    codebook(i,:) = lenna_training_set(i*32,:);
end

for times=1:10
    idx = zeros(16384,1);
    for i=1:16384
        iteration_result = realmax;
        for j=1:512
            Kmeans = sum((codebook(j,:)-lenna_training_set(i,:)).^2);
            if Kmeans < iteration_result
                idx(i,1) = j;
                iteration_result = Kmeans;
            end
        end
    end
    for m=1:512
        total = zeros(1,16);
        numbers = 0;
        for n=1:16384
            if idx(n,1) == m
                total = lenna_training_set(n,:) + total;
                numbers = numbers + 1;
            end
        end
        temp = total/numbers;
        if isnan(temp)
            codebook(m,:) = codebook(m,:);
        else    
            codebook(m,:) = temp;
        end
    end
end

for times=1:10
    idx = zeros(16384,1);
    for i=1:16384
        iteration_result = realmax;
        for j=1:512
            Kmeans = sum((codebook(j,:)-mandrill_training_set(i,:)).^2);
            if Kmeans < iteration_result
                idx(i,1) = j;
                iteration_result = Kmeans;
            end
        end
    end
    for m=1:512
        total = zeros(1,16);
        numbers = 0;
        for n=1:16384
            if idx(n,1) == m
                total = mandrill_training_set(n,:) + total;
                numbers = numbers + 1;
            end
        end
        temp = total/numbers;
        if isnan(temp)
            codebook(m,:) = codebook(m,:);
        else    
            codebook(m,:) = temp;
        end
    end
end

for times=1:10
    idx = zeros(16384,1);
    for i=1:16384
        iteration_result = realmax;
        for j=1:512
            Kmeans = sum((codebook(j,:)-scene_training_set(i,:)).^2);
            if Kmeans < iteration_result
                idx(i,1) = j;
                iteration_result = Kmeans;
            end
        end
    end
    for m=1:512
        total = zeros(1,16);
        numbers = 0;
        for n=1:16384
            if idx(n,1) == m
                total = scene_training_set(n,:) + total;
                numbers = numbers + 1;
            end
        end
        temp = total/numbers;
        if isnan(temp)
            codebook(m,:) = codebook(m,:);
        else    
            codebook(m,:) = temp;
        end
    end
end
for times=1:10
    idx = zeros(16384,1);
    for i=1:16384
        iteration_result = realmax;
        for j=1:512
            Kmeans = sum((codebook(j,:)-tiffany_training_set(i,:)).^2);
            if Kmeans < iteration_result
                idx(i,1) = j;
                iteration_result = Kmeans;
            end
        end
    end
    for m=1:512
        total = zeros(1,16);
        numbers = 0;
        for n=1:16384
            if idx(n,1) == m
                total = tiffany_training_set(n,:) + total;
                numbers = numbers + 1;
            end
        end
        temp = total/numbers;
        if isnan(temp)
            codebook(m,:) = codebook(m,:);
        else    
            codebook(m,:) = temp;
        end
    end
end
writematrix(codebook,'codebook.xls')
%{
%%reconstruct lenna to verify

reconstruct_lenna = zeros(512,152);
for i=1:16384
    start_pixel_row = floor((i-1)/128)*4 + 1;
    start_pixel_col = mod((i-1),128)*4 + 1;
    reconstruct_lenna(start_pixel_row,start_pixel_col:start_pixel_col+3) = lenna_training_set(i,1:4);
    reconstruct_lenna(start_pixel_row+1,start_pixel_col:start_pixel_col+3) = lenna_training_set(i,5:8);
    reconstruct_lenna(start_pixel_row+2,start_pixel_col:start_pixel_col+3) = lenna_training_set(i,9:12);
    reconstruct_lenna(start_pixel_row+3,start_pixel_col:start_pixel_col+3) = lenna_training_set(i,13:16);
end

disp("Lenna's PSNR: "+psnr(uint8(reconstruct_lenna),lenna));

%%reconstruct mandrill to verify

reconstruct_mandrill = zeros(512,152);
for i=1:16384
    start_pixel_row = floor((i-1)/128)*4 + 1;
    start_pixel_col = mod((i-1),128)*4 + 1;
    reconstruct_mandrill(start_pixel_row,start_pixel_col:start_pixel_col+3) = mandrill_training_set(i,1:4);
    reconstruct_mandrill(start_pixel_row+1,start_pixel_col:start_pixel_col+3) = mandrill_training_set(i,5:8);
    reconstruct_mandrill(start_pixel_row+2,start_pixel_col:start_pixel_col+3) = mandrill_training_set(i,9:12);
    reconstruct_mandrill(start_pixel_row+3,start_pixel_col:start_pixel_col+3) = mandrill_training_set(i,13:16);
end

disp("Mandrill's PSNR: "+psnr(uint8(reconstruct_mandrill),mandrill));

%%reconstruct scene to verify

reconstruct_scene = zeros(512,152);
for i=1:16384
    start_pixel_row = floor((i-1)/128)*4 + 1;
    start_pixel_col = mod((i-1),128)*4 + 1;
    reconstruct_scene(start_pixel_row,start_pixel_col:start_pixel_col+3) = scene_training_set(i,1:4);
    reconstruct_scene(start_pixel_row+1,start_pixel_col:start_pixel_col+3) = scene_training_set(i,5:8);
    reconstruct_scene(start_pixel_row+2,start_pixel_col:start_pixel_col+3) = scene_training_set(i,9:12);
    reconstruct_scene(start_pixel_row+3,start_pixel_col:start_pixel_col+3) = scene_training_set(i,13:16);
end

disp("Scene's PSNR: "+psnr(uint8(reconstruct_scene),scene));

%%reconstruct tiffany to verify

reconstruct_tiffany = zeros(512,152);
for i=1:16384
    start_pixel_row = floor((i-1)/128)*4 + 1;
    start_pixel_col = mod((i-1),128)*4 + 1;
    reconstruct_tiffany(start_pixel_row,start_pixel_col:start_pixel_col+3) = tiffany_training_set(i,1:4);
    reconstruct_tiffany(start_pixel_row+1,start_pixel_col:start_pixel_col+3) = tiffany_training_set(i,5:8);
    reconstruct_tiffany(start_pixel_row+2,start_pixel_col:start_pixel_col+3) = tiffany_training_set(i,9:12);
    reconstruct_tiffany(start_pixel_row+3,start_pixel_col:start_pixel_col+3) = tiffany_training_set(i,13:16);
end

disp("Tiffany's PSNR: "+psnr(uint8(reconstruct_tiffany),tiffany));
%}