function [x, box] = Read_Configuration(filename,varargin)

% Set frame limit if specified
if length(varargin) == 1
    frame_max = varargin{1};
else
    frame_max = inf;
end

% Open file
fid = fopen(filename);

% Get the number of particles
N = fscanf(fid, '%d\n', 1);
box = fscanf(fid, '%f %f %f\n', 3)';

% Initialization
line=''; %
frame = 1; % frame counter
x = []; % positions

% Loop through file in chunks
while ischar(line) && (frame <= frame_max)
    
    % Read the particle positions
    x_i = fscanf(fid, '%f %f %f\n', [3, N])';
    
    % Concatenate the current particle positions to the trajectory
    x = cat(3, x, x_i);
    
    % Increment frame counter
    frame = frame+1;
    
    % Skip N lines
    line = fgetl(fid); % N
    line = fgetl(fid); % box

end
    
% Close file
fclose(fid);

end