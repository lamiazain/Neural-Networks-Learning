function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
X=[ones(m,1) X] ;
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); %25x401
Theta2_grad = zeros(size(Theta2));%10x26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%          in ecomputedx4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%
a1=X; %5000x401
  z2 = X * Theta1';      % 5000x25
  a2=sigmoid(z2);        % 5000x25
  a2=[ones(m,1) a2] ;    %5000x26
  z3 = a2 * Theta2';     %5000x10
  a3=sigmoid(z3);        %5000x10
   
  h_x=a3;
  
y_new = zeros(m,num_labels );
for i=1:size(y,1)
    val=y(i);
   % y_new(i,:)=zeros(10,1)
    y_new(i,val)=1;
end
y_new;
Theta1_no_bias=Theta1(:,2:end);
Theta2_no_bias=Theta2(:,2:end);

reg_term = (lambda/(2*m)) * (sum(Theta1_no_bias(:).^2)+ sum(Theta2_no_bias(:).^2));
J = (1/m)*sum(sum((-y_new.*log(h_x))-((1-y_new).*log(1-h_x))))+reg_term;
  
%The gradient for the sigmoid function can be computed as
g_z2=a2.*( 1-a2)   ;    %5000x26

Delta3 = a3-y_new ;    %5000x10

Delta2 = (Delta3*Theta2).*g_z2; %5000x26
Delta2=Delta2(:,2:end);%5000x25

Theta1_grad = (1/m)*(Delta2'* a1); %25x401
Theta2_grad = (1/m)*(Delta3' *a2); %10x26
REG_term1 =  (1/m)*(lambda *  [Theta1(:,1)==0  Theta1(:,2:end)]) ;      %25x401
REG_term2 =  (1/m)*(lambda *  [Theta2(:,1)==0  Theta2(:,2:end)])  ;     %10x26
Theta1_grad = Theta1_grad + REG_term1;  %25x400
Theta2_grad = Theta2_grad + REG_term2;  %10x25



% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
