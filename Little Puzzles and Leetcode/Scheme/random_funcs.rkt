#lang racket

; -----
;  Predicates - iseven? or isodd?
(define (iseven? x)
  (= (modulo x 2) 0))

(define (isodd? x)
  (= (modulo x 2) 1))

(newline)
(display "Even Tests -> (iseven? atm) \n")
(iseven? 10)
(iseven? 21)
(iseven? 50)
(iseven? 97)
(newline)

(display "Odd Tests -> (isodd? atm) \n")
(isodd? 10)
(isodd? 21)
(isodd? 50)
(isodd? 97)
(newline)

; -----
;  Multiply all numbers in a list, including sublists.
;  [i.e.,   (product '(1 2 3) = 6 ]
(define (prod lst)
  (cond [(null? lst) 1]
    [(list? (first lst)) (* (prod (first lst)) (prod (rest lst)))]
    [(* (first lst) (prod (rest lst)))]))


(display "Product Tests -> (prod lst) \n")
(prod '(2 3 4 5))
(prod '(10 20 30))
(prod '(2 3 (2 3) 4 5))
(prod '(2 2 2 2))
(prod '(2 2 (2 2 (2 2 2) 2)))
(prod '())
(newline)

; -----
;  Multiply each element of a list by n.
;  [i.e.,   (lstmul 2 '(2 3 4)) = (list 4 6 8)]
(define (lstmul n lst)
  (cond [(null? lst) null]
        [(list? (first lst)) (cons (lstmul n (first lst)) (lstmul n (rest lst)))]
        [else (cons (* n (first lst)) (lstmul n (rest lst)))]))

(display "List Multiplication Tests -> (lstmul lst) \n")
(lstmul 2 '(2 3 4))
(lstmul 5 '(2 4 6 8))
(lstmul 2 '(1 2 (3 4)))
(lstmul 10 '(10 20 30 40))
(lstmul 2 '(100 200 (300 400)))
(lstmul 2 '(1 2 (3 4 (5 6) (7 8) 9) 10 (11) 12))
(lstmul 12 '())
(lstmul 10 '(10 20 (10 20) 30 40))
(lstmul 2 '(2 4 (2 4) 2 4 (2 2 (2 2)) 2 4))
(newline)

; -----
;  Sum numbers in a list, including sublists
;  [i.e.,   (sumlist '(2 3 4) = 9 ]

(define (sumlist lst)
  (cond [(null? lst) 0]
        [(list? (first lst)) (+ (sumlist (first lst)) (sumlist (rest lst)))]
        [else (+ (first lst) (sumlist (rest lst)))]))

(display "Sum List Tests -> (sumlist lst) \n")
(sumlist '(1 2 3 4 5 6 7 8 9))
(sumlist '(1 2 (2 3) 4))
(sumlist '(100 200 300 400 4 3 2 1))
(sumlist '(1 2 (3 4 (5 6 (7 8 (9 10) 11)) 12) 13 14))
(sumlist '())
(newline)

; -----
;  Length of a list.
;  [i.e.,   (len '(1 2 3 4 5 6 7 8)) = 8 ]
(define (len lst)
  (cond [(null? lst) 0]
        [(list? (first lst)) (+(len (first lst)) (len (rest lst)))]
        [else  (+ 1 (len (rest lst)))]))


(display "Length Tests -> (len lst) \n")
(len '(1 2 3 4 5 6 7 8))
(len '(4))
(len '(7 9 1 4 2))
(len '(7 9 (1 4) 2))
(len '(1 1 (1 1 (1 1 1) 1 ) 1 (1 1 1)))
(len '())
(newline)

; -----
;  Average of list.
;  [i.e.,   (average '(4 5 6 7 8)) = 6 ]

(define (average lst)
  (if (null? lst) 0
      (/ (sumlist lst) (len lst))))

(display "Average Tests -> (average lst) \n")
(average '(4 5 6 7 8))
(average '(1 (2) 3 (4 5 (6 7) 8 (9) 10)))
(average '(21 32 46 51 69))
(average '(111 222 333))
(average '(3))
(average '())
(average '(4 5 (6 7) 8))
(average '(2 2 (4 4) 2 (4 2 (4 2) 2 4) 4))
(newline)

; -----
;  Flatten the list
;  [i.e.,   (flatten '(1 2 (3 4 (5 6)))) = (list 1 2 3 4 5 6) ]

(define (flatten lst)
  (cond [(null? lst) null]
        [(list? (first lst)) (append (flatten (first lst)) (flatten (rest lst)))]
        [else (cons (first lst) (flatten (rest lst)))]))


(display "Flatten Lists Tests -> (flatten lst) \n")
(flatten '(1 2 (3 4 (5 6))))
(flatten '(1 2 (3 4 (5 6) 7 8 (9 10) 11 (12) 13 14) 15 16))
(flatten '(a ((c d) f) d))
(flatten '(a (((d f g)) e) h))
(flatten '())
(flatten '(1 1 (1 1 (1 1 1) 1 ) 1 (1 1 1)))
(newline)

; -----
;  Reverse all items in a list.
;  [i.e.,   (rvlst '(2 3 4 5)) = (list 5 4 3 2) ]

(define (rvlst lst)
  (cond [(null? lst) null]
        [(list? (first lst)) (append (rvlst (rest lst)) (list (rvlst (first lst))))]
        [else (append (rvlst (rest lst)) (list(first lst)))]))

(display "Reverse List Tests -> (revlst lst) \n")
(rvlst '(1 2 3 4 5 6))
(rvlst '(1 2 (3 4) (5 6) 7 8))
(rvlst '(9 87 6 5 4 3 2 1))
(rvlst '(10 20 30 40))
(rvlst '())
(rvlst '(9 8 (7 6 (5 4) 3) 2 1))
(newline)

; -----
;  Remove an item from a list.
;  [i.e.,   (rm 3 '(2 3 4 3)) = (list 2 4 3) ]

(define (rm n lst)
  (cond [(null? lst) null]
        [(list? (first lst)) (cons (rm n (first lst)) (rm n (rest lst)))]
        [(= n (first lst)) (rest lst)]
        [else (cons (first lst) (rm n (rest lst)))]))

(display "Remove Item from List Tests -> (rm lst) \n")
(rm 3 '(2 3 4 3))
(rm 18 '(12 14 87 12 18))
(rm 5 '(1 2 (3 4 (5) 6)))
(rm 3 '(2 3 4 3))
(rm 9 '(1 3 5 7 9 11 13))
(rm 3 '(4 7 (5 2) (8 1) (9 2 (3 1))))
(rm 8 '(0 2 (4 6 (8 10) 12) 14 16))
(rm 10 '())
(rm 50 '(10 (20 (30 (40 (50 60) 70) 80) 90) 100))
(newline)

; -----
;  Find smallest element in list.
;  Accepts list and item.
;  [i.e.,   (minimum '(7 5 (6 1))) = 1 ]

(define (minimum lst)
  (cond [(or (not (list? lst)) (null? lst)) lst] ; if lst is an atom then return the atom, or if it is null return null
        [(null? (rest lst)) (first lst)] ; return first element if only 1 element in list (the minimum)
        [else (minimum (cons {(lambda(x y) (if (< x y) x y)) (minimum(first lst)) (minimum(second lst))} (rest (rest lst))))]))
        ; else recursively call minimum on the modified list where the smallest number between the first 2 elements is first and the other number is removed.
        ; call minimum on first and second before testing in order to deal with case where first or second is list; this makes the number the minimum of the list.

(display "Minimum Item in List Tests -> (minimum lst) \n")
(minimum '(5 4 3 2 1))
(minimum '((2 3) 7 5 (6 1)))
(minimum '(5 2 7))
(minimum '(4121 3532 5522))
(minimum '((9 8) 1))
(minimum '(26 18 (27 21 19 (12 21 (7 10) 15) 17))) 
(minimum '(12 15 12 (71 34 (51 9)) 61))
(minimum '(21 15 (71 34 (51 (37 4)) 61) 32 41 18))
(minimum '())
(newline)

; -----
; Insertion Sort
;  [i.e.,   (insertion-sort '(9 1 8 2 7 3 6 4 5)) = (1 2 3 4 5 6 7 8 9) ]

(define (insertion-sort lst)
  (if [null? lst] null 
      (insert (first lst) (insertion-sort (rest lst))))) ; insert first element in rest of lst recursivly; practically inserts going backwards or last to first

(define (insert n lst) ; function to insert number n into lst in ordered spot
  (cond [(null? lst) (cons n lst)]
        [(> n (first lst)) (cons (first lst) (insert n (rest lst)))]
        [else (cons n lst)]))


(display "Insertion Sort Tests -> (insertion-sort lst) \n")
(insertion-sort '(5 9 8 4 2))
(insertion-sort '(5 9 2))
(insertion-sort '(5 9 1 2))
(insertion-sort '(12 7 3 5 9 11 1 8 10 4 2 6 12 1))
(newline)


; -----
;  Sqaure program.
;  Read a number from the user and output its value squared and cubed.
;  Note, uses built-in (read) function

(define (sqr-and-cube)
  (display "--------------------------------------------\n
Square and Cube Program.\n
Give me a number, and I'll compute its square and cube.\n\n")

  (let ((user-n (begin (display "Number: ")(read))))
      (newline) (display "The square of ") (display user-n) (display " is ")
      (display (* user-n user-n))
      (newline)
      (display "The cube of ") (display user-n) (display " is ")
      (display (* user-n user-n user-n)))
  )

(sqr-and-cube)

; -----
;  Simple list stats program.
;  Read a list from the user and compute the
;  length, sum, and average.
;  Note, also uses built-in (read) function
;  Also sorts lists and displays.

(define (liststats)
  (newline)(newline)(display "--------------------------------------------\n
List Stats Program.\n")

(let ((inLst (begin (display "List: ") (read))))
  (newline) (newline) (display "Length: ") (display (len inLst))
  (newline) (display "Average: ") (display (average inLst))
  (newline) (display "Minimum: ") (display (minimum inLst))
  (newline) (display "Sum: ") (display (sumlist inLst))
  (newline) (display "Product: ") (display (prod inLst))
  (newline)
  (newline) (display "Unsorted list:") (newline) (display inLst)
  (newline) (display "Sorted list:") (newline) (display (insertion-sort inLst)))
  )

(liststats)