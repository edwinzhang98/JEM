% 读取CSV文件
%data = csvread('1b.csv', 1, 0); % 跳过第一行，保留所有列

% 提取因变量和自变量
%dependent_variable = data(:, 3);
%independent_variables = data(:, 4:end);

% 灰色关联度分析函数
xxx=greyRelationalAnalysis(x, dependent_variable);
num_variables = size(independent_variables, 2);
correlation_matrix = zeros(num_variables, 1);
% 计算每个自变量与因变量的关联系数
for i = 1:num_variables
    x = independent_variables(:, i);
    correlation_matrix(i) = greyRelationalAnalysis(x, dependent_variable);
end

% 变量重要性排名
[sorted_correlation, rank] = sort(correlation_matrix, 'descend');

% 打印变量重要性排名
fprintf('变量重要性排名:\n');
for i = 1:num_variables
    fprintf('变量 %d: 相关系数 = %.4f\n', rank(i), sorted_correlation(i));
end

function result = greyRelationalAnalysis(x, y)
    % 定义灰色关联系数计算函数，这里使用绝对值差异法
    max_x = max(x);
    min_x = min(x);
    max_y = max(y);
    min_y = min(y);
    
    rho = 0.5;
    
    rx = (max_x - x + rho * (max_x - min_x)) / (max_x - min_x + rho * (max_x - min_x));
    ry = (max_y - y + rho * (max_y - min_y)) / (max_y - min_y + rho * (max_y - min_y));
    
    result = abs(rx - ry);
end
