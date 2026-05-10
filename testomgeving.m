load("DatasetCV.mat")
a1 = x;
a2 = y;
b = cat;

figure;
fouten = zeros(n,1);

for n=1:5
    A       = build_A(a1, a2, n);
    beta    = GD(A, b);
    b_hat   = classify(A, beta);
    fouten(n) = sum(b_hat ~= b);
    
    [xx, yy] = meshgrid(linspace(min(a1)-0.1, max(a1)+0.1, 500), ...
                        linspace(min(a2)-0.1, max(a2)+0.1, 500));
    A_grid = build_A(xx(:), yy(:), n);
    p_grid = 1 ./ (1 + exp(-A_grid * beta));
    p_grid = reshape(p_grid, size(xx));
    
    subplot(2, 3, n)
    hold on;
    grid on;
    xlabel('x1'); ylabel('x2')
    title(sprintf('n = %d', n));
    contour(xx, yy, p_grid, [0.5 0.5], 'k', 'LineWidth', 2);
    scatter(a1(b==1), a2(b==1), 20, 'b', 'filled');
    scatter(a1(b==-1), a2(b==-1), 20, 'r', 'filled');
    hold off
end

disp(fouten)