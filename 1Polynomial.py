class Polynomial: 
    def __init__(self, coefficients): 
        """ 
        Initialize a Polynomial object with a list of coefficients. 
        The coefficients are stored in increasing order of powers of x. 
        For example, [2, 3, 0, 5] represents 2 + 3x + 0x^2 + 5x^3. 
        """ 
        self.coefficients = coefficients 
 
    def order(self): 
        """ 
        Calculate and return the order (degree) of the polynomial. 
        The order is the highest power of x with a non-zero coefficient. 
        """ 
        return len(self.coefficients) - 1 
 
    def add(self, other): 
        """ 
        Add another Polynomial object to this one and return a new 
        Polynomial object. 
        The result accounts for cases where the polynomials have 
        different orders. 
        """ 
        # Determine the length of the resulting coefficients list 
        max_len = max(len(self.coefficients), len(other.coefficients)) 
        result = [0] * max_len  # Initialize the result coefficients with zeros 
 
        # Add corresponding coefficients from both polynomials 
        for i in range(len(self.coefficients)): 
            result[i] += self.coefficients[i] 
 
        for i in range(len(other.coefficients)): 
            result[i] += other.coefficients[i] 
 
        return Polynomial(result)  # Return a new Polynomial object with the resulting coefficients 
 
    def derivative(self): 
        """ 
        Calculate the derivative of the polynomial and return it as a new 
        Polynomial object. 
        The derivative of a polynomial is obtained by multiplying each 
        coefficient by its respective power and reducing the power by 1. 
        """ 
        # Special case: the derivative of a constant polynomial is 0 
        if len(self.coefficients) == 1: 
            return Polynomial([0]) 
 
        # Calculate the derivative coefficients 
        result = [(i * coef) for i, coef in enumerate(self.coefficients)][1:] 

        return Polynomial(result) 
 
    def antiderivative(self, constant): 
        """ 
        Calculate the indefinite integral (antiderivative) of the 
        polynomial and return it as a new Polynomial object. 
        A constant of integration is added as the first term. 
        """ 
        # Initialize the result with an extra term for the constant of integration 
        result = [0] * (len(self.coefficients) + 1) 
        result[0] = constant  # Set the constant of integration 
 
        # Compute the integral coefficients by dividing each coefficient by its new power 
        for i, coef in enumerate(self.coefficients): 
            result[i + 1] = coef / (i + 1) 
 
        return Polynomial(result)  # Return a new Polynomial object with the integral coefficients 
 
    def __str__(self): 
        """ 
        Create a human-readable string representation of the polynomial. 
        The format is: P(x) = a_0 + a_1 x + a_2 x^2 + ... + a_n x^n. 
        """ 
        terms = [] 
        for i, coef in enumerate(self.coefficients): 
            if coef != 0:  # Skip terms with a coefficient of 0 
                if i == 0:  # Constant term (x^0) 
                    terms.append(f"{coef}") 
                elif i == 1:  # Linear term (x^1) 
                    terms.append(f"{coef}x") 
                else:  # Higher-order terms (x^2 and above) 
                    terms.append(f"{coef}x^{i}") 
        return " + ".join(terms) if terms else "0"  # Return "0" if all coefficients are zero 
 
 
# Testing the class functionality 
if __name__ == "__main__": 
    # Define two Polynomial objects 
    Pa = Polynomial([2, 0, 4, -1, 0, 6])  # P_a(x) = 2 + 4x^2 - x^3 + 6x^5 
    Pb = Polynomial([-1, -3, 0, 4.5])     # P_b(x) = -1 - 3x + 4.5x^3 
 
    # Print the polynomials 
    print("P_a(x):", Pa) 
    print("P_b(x):", Pb) 
 
    # Calculate and print the order of P_a(x) 
    print("Order of P_a(x):", Pa.order()) 
 
    # Add P_b(x) to P_a(x) and print the result 
    P_sum = Pa.add(Pb) 
    print("P_a(x) + P_b(x):", P_sum) 
 
    # Calculate and print the derivative of P_a(x) 
    P_derivative = Pa.derivative() 
    print("Derivative of P_a(x):", P_derivative) 
 
    # Calculate and print the antiderivative of the derivative of P_a(x), with c=2 
    P_antiderivative = P_derivative.antiderivative(2) 
    print("Antiderivative of P_a'(x) with c=2:", P_antiderivative) 
