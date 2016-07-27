function [ v ] = gen_steering_vectors( Ron, Roff )
%GEN_STEERING_VECTORS Estimates array steering vectors based on
%on/off-source array covariance matrices
%   Detailed explanation goes here

    % Make sure the matrices are the same size
    if sum(size(Ron) == size(Roff)) ~= 3
        error(['Covariance matrices must be the same size!\n',...
               '\t size(Ron)  = [', num2str(size(Ron)),  ']\n',...
               '\t size(Roff) = [', num2str(size(Roff)), ']']);
    end
    
    % Make sure that the matrices are square
    if size(Ron, 1) ~= size(Ron, 2)
        error(['Covariance matrices must be square!\n',...
               '\t size(Ron) = [', num2str(size(Ron)), ']']);
    end
    
    % Extract dimensions
    Nele = size(Ron, 1);
    Nbin = size(Ron, 3);
    
    % Verify that the off matrix is full-rank
    offranks = zeros(Nbin, 1);
    for b = 1:Nbin
        offranks(b) = rank(Roff(:,:,b));
    end
    num_sing_off = sum(offranks == 0);
    if num_sing_off > 0
        error('Off-source Covariance matrix must be full rank!\n');
    end
    
    % Compute steering vectors
    v = zeros(Nele, Nbin);
    for b = 1:Nbin
        [vtmp,~] = eigs(Ron(:,:,b), Roff(:,:,b), 1);
        v(:,b) = vtmp./norm(vtmp);
    end

end

