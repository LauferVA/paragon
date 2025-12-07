# Fibonacci Calculator

A simple web API that calculates Fibonacci numbers.

## Overview

Build a REST API service that can calculate Fibonacci numbers efficiently. The service should handle large numbers and provide both iterative and recursive implementations.

## Target User

Developers who need a reliable Fibonacci calculation service for testing, education, or integration into larger applications.

## Must-Have Features

- Calculate Fibonacci numbers up to n=1000
- Support both GET and POST requests
- Return results in JSON format
- Include timing information for performance analysis
- Provide both iterative and recursive calculation methods

## Technical Details

**Stack**: Python, FastAPI
**Database**: None required (stateless service)
**Performance**: Response time < 100ms for n <= 100

## Requirements

1. Endpoint `/fib/{n}` returns the nth Fibonacci number
2. Endpoint `/fib/batch` accepts array of numbers and returns all results
3. Support query parameter `?method=iterative|recursive`
4. Include request timing in response
5. Handle errors gracefully (negative numbers, overflow, etc.)

## Success Criteria

- All endpoints return correct Fibonacci numbers
- Response time < 100ms for n <= 100
- Proper error handling with meaningful error messages
- 100% test coverage for core calculation logic
