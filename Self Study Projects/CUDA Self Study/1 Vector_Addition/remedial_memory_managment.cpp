//
// Created by jwens on 7/5/2025.
// I'm a spoiled little Python brat
// Practice file to get a better grasp of pointers before jumping into CUDA

// =======================================================
// Section 1: Simple pointer Deomnstration
// =======================================================

// #include <iostream>
//
// int main() {
//     int x = 42;                     // A regular integer variable
//     int* ptr = &x;                  // A pointer that holds the address of x
//
//     std::cout << "Value of x: " << x << '\n';
//     std::cout << "Address of x (&x): " << &x << '\n';
//     std::cout << "Value of ptr (should match &x): " << ptr << '\n';
//     std::cout << "Value pointed to by ptr (*ptr): " << *ptr << '\n';
//
//
//     // Now let's change x using the pointer
//     *ptr = 99;
//     std::cout << "New value of x (after *ptr = 99): " << x << '\n';
//
//     // Let's test it out
//     std::cout << "New value of x = " << *ptr << '\n';
//
//     return 0;
// }

// - Declares an integer variable `x` and assigns it the value 42.
// - Declares a pointer to int named `ptr` and stores the address of `x` in it.
// - Prints the value of `x`.
// - Prints the memory address of `x` using `&x`.
// - Prints the value of `ptr`, which holds the same address as `&x`.
// - Prints the value at the memory location pointed to by `ptr` using `*ptr`.
// - Updates the value at the memory location `ptr` points to by assigning `*ptr = 99`.
// - Prints the new value of `x`, confirming that it was modified through the pointer.

// =======================================================
// Section 2: Dynamic Memory Allocation (Heap)
// =======================================================

// Dynamic Memory Allocation
// #include <iostream>
//
// // Function to demonstrate dynamic allocation on the heap
// void dynamic_allocation_demo() {
//     std::cout << "\n=== Dynamic Memory Allocation Demo ===\n";
//
//     int* heap_var = new int;       // Allocate memory on the heap
//     *heap_var = 1337;              // Store a value in that heap memory
//
//     std::cout << "heap_var points to address: " << heap_var << '\n';
//     std::cout << "Value at heap_var (*heap_var): " << *heap_var << '\n';
//
//     delete heap_var;               // Free the heap memory
//     heap_var = nullptr;            // Null the pointer to avoid dangling reference
//
//     std::cout << "heap_var deleted and set to nullptr.\n";
// }
//
// int main() {
//     dynamic_allocation_demo();
//     return 0;
// }

// - Declares a pointer to an integer named `heap_var`.
// - Allocates memory for a single int on the heap using `new`.
// - Stores the value 1337 in the heap-allocated memory using `*heap_var = 1337`.
// - Prints the memory address that `heap_var` points to.
// - Prints the value stored at that memory address using `*heap_var`.
// - Frees the heap memory using `delete heap_var`.
// - Sets `heap_var` to `nullptr` to avoid a dangling pointer.
// - Confirms that the pointer was reset by printing a message.

// =======================================================
// Section 3: Pointer Arithmetic
// =======================================================

// Pointer Arithmetic
#include <iostream>

// Function to demonstrate pointer arithmetic on a dynamic array
void pointer_arithmetic_demo() {
    std::cout << "\n=== Pointer Arithmetic Demo ===\n";

    int size = 5;                                  // - Declare the number of elements in the array
    int* arr = new int[size];                      // - Allocate a dynamic array of integers on the heap

    // - Populate the array using pointer arithmetic
    for (int i = 0; i < size; ++i) {
        *(arr + i) = (i + 1) * 10;                 // - Set value at arr[i] using pointer math
    }

    // - Print the array values using pointer arithmetic
    for (int i = 0; i < size; ++i) {
        std::cout << "Element " << i << " = " << *(arr + i) << '\n';
    }

    delete[] arr;                                  // - Free the memory allocated for the dynamic array
    arr = nullptr;                                 // - Null the pointer to avoid dangling references

    std::cout << "Array memory released and pointer nulled.\n";
}

int main() {
    pointer_arithmetic_demo();
    return 0;
}

// - Declares a variable `size` for the number of integers we want to store.
// - Uses `new int[size]` to allocate an array of integers on the heap.
// - Uses pointer arithmetic: `*(arr + i)` is equivalent to `arr[i]`.
// - Stores values like 10, 20, 30, ... into the array.
// - Prints out those values by moving the pointer using `(arr + i)`.
// - Calls `delete[]` to clean up the heap memory.
// - Sets the pointer to `nullptr` to prevent accidental reuse.

