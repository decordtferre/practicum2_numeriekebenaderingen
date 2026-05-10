function [beta, step] = GD(A, b)

    [N, n] = size(A);
    beta   = zeros(n, 1);
    step   = 0.01;
    tol    = 1e-6;

    for k = 1:10000
        p    = 1 ./ (1 + exp(-b .* (A * beta)));
        grad = -(1/N) * (A' * ((1 - p) .* b));
        
        % vroeger stoppen als als de gradient reeds klein genoeg is
        if norm(grad) < tol
            break
        end

        beta = beta - step * grad;
    end
end