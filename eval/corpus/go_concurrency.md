# Goroutines
A goroutine is a lightweight thread managed by the Go runtime. Starting one costs only a few kilobytes of stack, so a program can run hundreds of thousands of them. You start a goroutine by putting the keyword go in front of a function call.

# Channels
Channels are typed pipes that let goroutines send values to each other. An unbuffered channel blocks the sender until a receiver is ready, which makes it a synchronisation point. A buffered channel only blocks when its buffer is full.

# Select
The select statement waits on several channel operations at once and runs the first one that is ready. A default case makes select non-blocking.

# WaitGroup
A sync.WaitGroup waits for a collection of goroutines to finish. Call Add before starting each goroutine, Done when it finishes, and Wait to block until the counter reaches zero.

# Mutex
A sync.Mutex protects shared memory so only one goroutine at a time can change it. Always unlock with defer right after locking so a panic cannot leave the mutex locked.
