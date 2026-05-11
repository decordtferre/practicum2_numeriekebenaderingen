load("DatasetCV.mat")
a1 = x;
a2 = y;
b = cat;
max_n = 20;

xr = zeros(0, 0);
yr = zeros(0, 0);
catr = zeros(0, 0);

xe = zeros(0, 0);
ye = zeros(0, 0);
cate = zeros(0, 0);

% Verdeling opstellen
for i = 1:length(a1)
    if randi([0 1]) == 1
        xr = [xr; x(i)];
        yr = [yr; y(i)];
        catr = [catr; cat(i)];
    else
        xe = [xe; x(i)];
        ye = [ye; y(i)];
        cate = [cate; cat(i)];
    end
end

% Trainen en testen
for n=1:max_n
    %Trainen model
    A       = build_A(xr, yr, n);
    beta    = GD(A, catr);

    %Testdata evalueren
    A_e = build_A(xe, ye, n);
    b_hat   = classify(A_e, beta);
    fouten(n) = sum(b_hat ~= cate);
    
    [xx, yy] = meshgrid(linspace(min(a1)-0.1, max(a1)+0.1, 500), ...
                        linspace(min(a2)-0.1, max(a2)+0.1, 500));
    A_grid = build_A(xx(:), yy(:), n);
    p_grid = 1 ./ (1 + exp(-A_grid * beta));
    p_grid = reshape(p_grid, size(xx));
    
    subplot(4, 5, n)
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