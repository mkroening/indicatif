use std::borrow::Cow;
use std::fmt;
use std::io;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Weak};
use std::time::{Duration, Instant};

use crate::state::{
    MultiObject, MultiProgressState, ProgressDrawState, ProgressDrawTarget, ProgressDrawTargetKind,
    ProgressState, Status,
};
use crate::style::ProgressStyle;
use crate::utils::Estimate;
use async_channel::{Receiver, Sender};
use async_lock::{Mutex, RwLock};

/// A progress bar or spinner
///
/// The progress bar is an [`Arc`] around its internal state. When the progress bar is cloned it
/// just increments the refcount (so the original and its clone share the same state).
#[derive(Clone)]
pub struct ProgressBar {
    state: Arc<Mutex<ProgressState>>,
}

impl fmt::Debug for ProgressBar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("ProgressBar").finish()
    }
}

impl ProgressBar {
    /// Creates a new progress bar with a given length
    ///
    /// This progress bar by default draws directly to stderr, and refreshes a maximum of 15 times
    /// a second. To change the refresh rate, set the draw target to one with a different refresh
    /// rate.
    pub fn new(len: u64) -> ProgressBar {
        ProgressBar::with_draw_target(len, ProgressDrawTarget::stderr())
    }

    /// Creates a completely hidden progress bar
    ///
    /// This progress bar still responds to API changes but it does not have a length or render in
    /// any way.
    pub fn hidden() -> ProgressBar {
        ProgressBar::with_draw_target(!0, ProgressDrawTarget::hidden())
    }

    /// Creates a new progress bar with a given length and draw target
    pub fn with_draw_target(len: u64, target: ProgressDrawTarget) -> ProgressBar {
        ProgressBar {
            state: Arc::new(Mutex::new(ProgressState {
                style: ProgressStyle::default_bar(),
                draw_target: target,
                message: "".into(),
                prefix: "".into(),
                pos: 0,
                len,
                tick: 0,
                draw_delta: 0,
                draw_rate: 0,
                draw_next: 0,
                status: Status::InProgress,
                started: Instant::now(),
                est: Estimate::new(),
                steady_tick: 0,
            })),
        }
    }

    /// A convenience builder-like function for a progress bar with a given style
    pub async fn with_style(self, style: ProgressStyle) -> ProgressBar {
        self.state.lock().await.style = style;
        self
    }

    /// A convenience builder-like function for a progress bar with a given prefix
    pub async fn with_prefix(self, prefix: impl Into<Cow<'static, str>>) -> ProgressBar {
        self.state.lock().await.prefix = prefix.into();
        self
    }

    /// A convenience builder-like function for a progress bar with a given message
    pub async fn with_message(self, message: impl Into<Cow<'static, str>>) -> ProgressBar {
        self.state.lock().await.message = message.into();
        self
    }

    /// A convenience builder-like function for a progress bar with a given position
    pub async fn with_position(self, pos: u64) -> ProgressBar {
        self.state.lock().await.pos = pos;
        self
    }

    /// Creates a new spinner
    ///
    /// This spinner by default draws directly to stderr. This adds the default spinner style to it.
    pub async fn new_spinner() -> ProgressBar {
        let rv = ProgressBar::new(!0);
        rv.set_style(ProgressStyle::default_spinner()).await;
        rv
    }

    /// Overrides the stored style
    ///
    /// This does not redraw the bar. Call [`ProgressBar::tick()`] to force it.
    pub async fn set_style(&self, style: ProgressStyle) {
        self.state.lock().await.style = style;
    }

    /// Limit redrawing of progress bar to every `n` steps
    ///
    /// By default, the progress bar will redraw whenever its state advances. This setting is
    /// helpful in situations where the overhead of redrawing the progress bar dominates the
    /// computation whose progress is being reported.
    ///
    /// If `n` is greater than 0, operations that change the progress bar such as
    /// [`ProgressBar::tick()`], [`ProgressBar::set_message()`] and [`ProgressBar::set_length()`]
    /// will no longer cause the progress bar to be redrawn, and will only be shown once the
    /// position advances by `n` steps.
    ///
    /// ```rust,no_run
    /// # use indicatif::ProgressBar;
    /// let n = 1_000_000;
    /// let pb = ProgressBar::new(n);
    /// pb.set_draw_delta(n / 100); // redraw every 1% of additional progress
    /// ```
    ///
    /// Note that `ProgressDrawTarget` may impose additional buffering of redraws.
    pub async fn set_draw_delta(&self, n: u64) {
        let mut state = self.state.lock().await;
        state.draw_delta = n;
        state.draw_next = state.pos.saturating_add(state.draw_delta);
    }

    /// Sets the refresh rate of progress bar to `n` updates per seconds
    ///
    /// This is similar to `set_draw_delta` but automatically adapts to a constant refresh rate
    /// regardless of how consistent the progress is.
    ///
    /// This parameter takes precedence on `set_draw_delta` if different from 0.
    ///
    /// ```rust,no_run
    /// # use indicatif::ProgressBar;
    /// let n = 1_000_000;
    /// let pb = ProgressBar::new(n);
    /// pb.set_draw_rate(25); // aims at redrawing at most 25 times per seconds.
    /// ```
    ///
    /// Note that the [`ProgressDrawTarget`] may impose additional buffering of redraws.
    pub async fn set_draw_rate(&self, n: u64) {
        let mut state = self.state.lock().await;
        state.draw_rate = n;
        state.draw_next = state.pos.saturating_add(state.per_sec() / n);
    }

    /// Manually ticks the spinner or progress bar
    ///
    /// This automatically happens on any other change to a progress bar.
    pub async fn tick(&self) {
        self.update_and_draw(|state| {
            if state.steady_tick == 0 || state.tick == 0 {
                state.tick = state.tick.saturating_add(1);
            }
        })
        .await;
    }

    /// Advances the position of the progress bar by `delta`
    pub async fn inc(&self, delta: u64) {
        self.update_and_draw(|state| {
            state.pos = state.pos.saturating_add(delta);
            if state.steady_tick == 0 || state.tick == 0 {
                state.tick = state.tick.saturating_add(1);
            }
        })
        .await
    }

    /// A quick convenience check if the progress bar is hidden
    pub async fn is_hidden(&self) -> bool {
        self.state.lock().await.draw_target.is_hidden()
    }

    /// Indicates that the progress bar finished
    pub async fn is_finished(&self) -> bool {
        self.state.lock().await.is_finished()
    }

    /// Print a log line above the progress bar
    ///
    /// If the progress bar was added to a [`MultiProgress`], the log line will be
    /// printed above all other progress bars.
    ///
    /// Note that if the progress bar is hidden (which by default happens if the progress bar is
    /// redirected into a file) `println()` will not do anything either.
    pub async fn println<I: AsRef<str>>(&self, msg: I) {
        let mut state = self.state.lock().await;

        let mut lines: Vec<String> = msg.as_ref().lines().map(Into::into).collect();
        let orphan_lines = lines.len();
        if state.should_render() && !state.draw_target.is_hidden() {
            lines.extend(state.style.format_state(&*state).await);
        }

        let draw_state = ProgressDrawState {
            lines,
            orphan_lines,
            finished: state.is_finished(),
            force_draw: true,
            move_cursor: false,
        };

        state.draw_target.apply_draw_state(draw_state).await.ok();
    }

    /// Sets the position of the progress bar
    pub async fn set_position(&self, pos: u64) {
        self.update_and_draw(|state| {
            state.draw_next = pos;
            state.pos = pos;
            if state.steady_tick == 0 || state.tick == 0 {
                state.tick = state.tick.saturating_add(1);
            }
        })
        .await
    }

    /// Sets the length of the progress bar
    pub async fn set_length(&self, len: u64) {
        self.update_and_draw(|state| {
            state.len = len;
        })
        .await
    }

    /// Increase the length of the progress bar
    pub async fn inc_length(&self, delta: u64) {
        self.update_and_draw(|state| {
            state.len = state.len.saturating_add(delta);
        })
        .await
    }

    /// Sets the current prefix of the progress bar
    ///
    /// For the prefix to be visible, the `{prefix}` placeholder must be present in the template
    /// (see [`ProgressStyle`]).
    pub async fn set_prefix(&self, prefix: impl Into<Cow<'static, str>>) {
        let prefix = prefix.into();
        self.update_and_draw(|state| {
            state.prefix = prefix;
            if state.steady_tick == 0 || state.tick == 0 {
                state.tick = state.tick.saturating_add(1);
            }
        })
        .await
    }

    /// Sets the current message of the progress bar
    ///
    /// For the message to be visible, the `{msg}` placeholder must be present in the template (see
    /// [`ProgressStyle`]).
    pub async fn set_message(&self, msg: impl Into<Cow<'static, str>>) {
        let msg = msg.into();
        self.update_and_draw(|state| {
            state.message = msg;
            if state.steady_tick == 0 || state.tick == 0 {
                state.tick = state.tick.saturating_add(1);
            }
        })
        .await
    }

    /// Creates a new weak reference to this `ProgressBar`
    pub fn downgrade(&self) -> WeakProgressBar {
        WeakProgressBar {
            state: Arc::downgrade(&self.state),
        }
    }

    /// Resets the ETA calculation
    ///
    /// This can be useful if the progress bars made a large jump or was paused for a prolonged
    /// time.
    pub async fn reset_eta(&self) {
        self.update_and_draw(|state| {
            state.est.reset(state.pos);
        })
        .await;
    }

    /// Resets elapsed time
    pub async fn reset_elapsed(&self) {
        self.update_and_draw(|state| {
            state.started = Instant::now();
        })
        .await;
    }

    /// Resets all of the progress bar state
    pub async fn reset(&self) {
        self.reset_eta().await;
        self.reset_elapsed().await;
        self.update_and_draw(|state| {
            state.draw_next = 0;
            state.pos = 0;
            state.status = Status::InProgress;
        })
        .await;
    }

    /// Finishes the progress bar and leaves the current message
    pub async fn finish(&self) {
        self.state.lock().await.finish().await;
    }

    /// Finishes the progress bar at current position and leaves the current message
    pub async fn finish_at_current_pos(&self) {
        self.state.lock().await.finish_at_current_pos().await;
    }

    /// Finishes the progress bar and sets a message
    ///
    /// For the message to be visible, the `{msg}` placeholder must be present in the template (see
    /// [`ProgressStyle`]).
    pub async fn finish_with_message(&self, msg: impl Into<Cow<'static, str>>) {
        self.state.lock().await.finish_with_message(msg).await;
    }

    /// Finishes the progress bar and completely clears it
    pub async fn finish_and_clear(&self) {
        self.state.lock().await.finish_and_clear().await;
    }

    /// Finishes the progress bar and leaves the current message and progress
    pub async fn abandon(&self) {
        self.state.lock().await.abandon().await;
    }

    /// Finishes the progress bar and sets a message, and leaves the current progress
    ///
    /// For the message to be visible, the `{msg}` placeholder must be present in the template (see
    /// [`ProgressStyle`]).
    pub async fn abandon_with_message(&self, msg: impl Into<Cow<'static, str>>) {
        self.state.lock().await.abandon_with_message(msg).await;
    }

    /// Finishes the progress bar using the behavior stored in the [`ProgressStyle`]
    ///
    /// See [`ProgressStyle::on_finish()`].
    pub async fn finish_using_style(&self) {
        self.state.lock().await.finish_using_style().await;
    }

    /// Sets a different draw target for the progress bar
    ///
    /// This can be used to draw the progress bar to stderr (this is the default):
    ///
    /// ```rust,no_run
    /// # use indicatif::{ProgressBar, ProgressDrawTarget};
    /// let pb = ProgressBar::new(100);
    /// pb.set_draw_target(ProgressDrawTarget::stderr());
    /// ```
    ///
    /// **Note:** Calling this method on a [`ProgressBar`] linked with a [`MultiProgress`] (after
    /// running [`MultiProgress::add`]) will unlink this progress bar. If you don't want this
    /// behavior, call [`MultiProgress::set_draw_target`] instead.
    pub async fn set_draw_target(&self, target: ProgressDrawTarget) {
        let mut state = self.state.lock().await;
        state.draw_target.disconnect().await;
        state.draw_target = target;
    }

    async fn update_and_draw<F: FnOnce(&mut ProgressState)>(&self, f: F) {
        // Delegate to the wrapped state.
        let mut state = self.state.lock().await;
        state.update_and_draw(f).await;
    }

    /// Returns the current position
    pub async fn position(&self) -> u64 {
        self.state.lock().await.pos
    }

    /// Returns the current length
    pub async fn length(&self) -> u64 {
        self.state.lock().await.len
    }

    /// Returns the current ETA
    pub async fn eta(&self) -> Duration {
        self.state.lock().await.eta()
    }

    /// Returns the current rate of progress
    pub async fn per_sec(&self) -> u64 {
        self.state.lock().await.per_sec()
    }

    /// Returns the current expected duration
    pub async fn duration(&self) -> Duration {
        self.state.lock().await.duration()
    }

    /// Returns the current elapsed time
    pub async fn elapsed(&self) -> Duration {
        self.state.lock().await.started.elapsed()
    }
}

/// Manages multiple progress bars from different threads
pub struct MultiProgress {
    state: Arc<RwLock<MultiProgressState>>,
    joining: AtomicBool,
    tx: Sender<(usize, ProgressDrawState)>,
    rx: Receiver<(usize, ProgressDrawState)>,
}

impl fmt::Debug for MultiProgress {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("MultiProgress").finish()
    }
}

unsafe impl Sync for MultiProgress {}

impl Default for MultiProgress {
    fn default() -> MultiProgress {
        MultiProgress::with_draw_target(ProgressDrawTarget::stderr())
    }
}

impl MultiProgress {
    /// Creates a new multi progress object.
    ///
    /// Progress bars added to this object by default draw directly to stderr, and refresh
    /// a maximum of 15 times a second. To change the refresh rate set the draw target to
    /// one with a different refresh rate.
    pub fn new() -> MultiProgress {
        MultiProgress::default()
    }

    /// Creates a new multi progress object with the given draw target.
    pub fn with_draw_target(draw_target: ProgressDrawTarget) -> MultiProgress {
        let (tx, rx) = async_channel::unbounded();
        MultiProgress {
            state: Arc::new(RwLock::new(MultiProgressState {
                objects: Vec::new(),
                free_set: Vec::new(),
                ordering: vec![],
                draw_target,
                move_cursor: false,
            })),
            joining: AtomicBool::new(false),
            tx,
            rx,
        }
    }

    /// Sets a different draw target for the multiprogress bar.
    pub async fn set_draw_target(&self, target: ProgressDrawTarget) {
        let mut state = self.state.write().await;
        state.draw_target.disconnect().await;
        state.draw_target = target;
    }

    /// Set whether we should try to move the cursor when possible instead of clearing lines.
    ///
    /// This can reduce flickering, but do not enable it if you intend to change the number of
    /// progress bars.
    pub async fn set_move_cursor(&self, move_cursor: bool) {
        self.state.write().await.move_cursor = move_cursor;
    }

    /// Adds a progress bar.
    ///
    /// The progress bar added will have the draw target changed to a
    /// remote draw target that is intercepted by the multi progress
    /// object overriding custom `ProgressDrawTarget` settings.
    pub async fn add(&self, pb: ProgressBar) -> ProgressBar {
        self.push(None, pb).await
    }

    /// Inserts a progress bar.
    ///
    /// The progress bar inserted at position `index` will have the draw
    /// target changed to a remote draw target that is intercepted by the
    /// multi progress object overriding custom `ProgressDrawTarget` settings.
    ///
    /// If `index >= MultiProgressState::objects.len()`, the progress bar
    /// is added to the end of the list.
    pub async fn insert(&self, index: usize, pb: ProgressBar) -> ProgressBar {
        self.push(Some(index), pb).await
    }

    async fn push(&self, pos: Option<usize>, pb: ProgressBar) -> ProgressBar {
        let new = MultiObject {
            done: false,
            draw_state: None,
        };

        let mut state = self.state.write().await;
        let idx = match state.free_set.pop() {
            Some(idx) => {
                state.objects[idx] = Some(new);
                idx
            }
            None => {
                state.objects.push(Some(new));
                state.objects.len() - 1
            }
        };

        match pos {
            Some(pos) if pos < state.ordering.len() => state.ordering.insert(pos, idx),
            _ => state.ordering.push(idx),
        }

        pb.set_draw_target(ProgressDrawTarget {
            kind: ProgressDrawTargetKind::Remote {
                state: self.state.clone(),
                idx,
                chan: Mutex::new(self.tx.clone()),
            },
        })
        .await;
        pb
    }

    /// Removes a progress bar.
    ///
    /// The progress bar is removed only if it was previously inserted or added
    /// by the methods `MultiProgress::insert` or `MultiProgress::add`.
    /// If the passed progress bar does not satisfy the condition above,
    /// the `remove` method does nothing.
    pub async fn remove(&self, pb: &ProgressBar) {
        let idx = match &pb.state.lock().await.draw_target.kind {
            ProgressDrawTargetKind::Remote { state, idx, .. } => {
                // Check that this progress bar is owned by the current MultiProgress.
                assert!(Arc::ptr_eq(&self.state, state));
                *idx
            }
            _ => return,
        };

        self.state.write().await.remove_idx(idx);
    }

    /// Waits for all progress bars to report that they are finished.
    ///
    /// You need to call this as this will request the draw instructions
    /// from the remote progress bars.  Not calling this will deadlock
    /// your program.
    pub async fn join(&self) -> io::Result<()> {
        self.join_impl(false).await
    }

    /// Works like `join` but clears the progress bar in the end.
    pub async fn join_and_clear(&self) -> io::Result<()> {
        self.join_impl(true).await
    }

    async fn join_impl(&self, clear: bool) -> io::Result<()> {
        if self.joining.load(Ordering::Acquire) {
            panic!("Already joining!");
        }
        self.joining.store(true, Ordering::Release);

        let move_cursor = self.state.read().await.move_cursor;
        // Max amount of grouped together updates at once. This is meant
        // to ensure there isn't a situation where continuous updates prevent
        // any actual draws happening.
        const MAX_GROUP_SIZE: usize = 32;
        let mut recv_peek = None;
        let mut grouped = 0usize;
        let mut orphan_lines: Vec<String> = Vec::new();
        let mut force_draw = false;
        while !self.state.read().await.is_done() {
            let (idx, draw_state) = if let Some(peeked) = recv_peek.take() {
                peeked
            } else {
                self.rx.recv().await.unwrap()
            };
            force_draw |= draw_state.finished || draw_state.force_draw;

            let mut state = self.state.write().await;
            if draw_state.finished {
                if let Some(ref mut obj) = &mut state.objects[idx] {
                    obj.done = true;
                }
                if draw_state.lines.is_empty() {
                    // `finish_and_clear` was called
                    state.remove_idx(idx);
                }
            }

            // Split orphan lines out of the draw state, if any
            let lines = if draw_state.orphan_lines > 0 {
                let split = draw_state.lines.split_at(draw_state.orphan_lines);
                orphan_lines.extend_from_slice(split.0);
                split.1.to_vec()
            } else {
                draw_state.lines
            };

            let draw_state = ProgressDrawState {
                lines,
                orphan_lines: 0,
                ..draw_state
            };

            if let Some(ref mut obj) = &mut state.objects[idx] {
                obj.draw_state = Some(draw_state);
            }

            // the rest from here is only drawing, we can skip it.
            if state.draw_target.is_hidden() {
                continue;
            }

            debug_assert!(recv_peek.is_none());
            if grouped >= MAX_GROUP_SIZE {
                // Can't group any more draw calls, proceed to just draw
                grouped = 0;
            } else if let Ok(state) = self.rx.try_recv() {
                // Only group draw calls if there is another draw already queued
                recv_peek = Some(state);
                grouped += 1;
                continue;
            } else {
                // No more draws queued, proceed to just draw
                grouped = 0;
            }

            let mut lines = vec![];

            // Make orphaned lines appear at the top, so they can be properly
            // forgotten.
            let orphan_lines_count = orphan_lines.len();
            lines.append(&mut orphan_lines);

            for index in state.ordering.iter() {
                if let Some(obj) = &state.objects[*index] {
                    if let Some(ref draw_state) = obj.draw_state {
                        lines.extend_from_slice(&draw_state.lines[..]);
                    }
                }
            }

            let finished = state.is_done();
            state
                .draw_target
                .apply_draw_state(ProgressDrawState {
                    lines,
                    orphan_lines: orphan_lines_count,
                    force_draw: force_draw || orphan_lines_count > 0,
                    move_cursor,
                    finished,
                })
                .await?;

            force_draw = false;
        }

        if clear {
            let mut state = self.state.write().await;
            state
                .draw_target
                .apply_draw_state(ProgressDrawState {
                    lines: vec![],
                    orphan_lines: 0,
                    finished: true,
                    force_draw: true,
                    move_cursor,
                })
                .await?;
        }

        self.joining.store(false, Ordering::Release);

        Ok(())
    }
}

/// A weak reference to a `ProgressBar`.
///
/// Useful for creating custom steady tick implementations
#[derive(Clone)]
pub struct WeakProgressBar {
    state: Weak<Mutex<ProgressState>>,
}

impl WeakProgressBar {
    /// Attempts to upgrade the Weak pointer to a [`ProgressBar`], delaying dropping of the inner
    /// value if successful. Returns `None` if the inner value has since been dropped.
    ///
    /// [`ProgressBar`]: struct.ProgressBar.html
    pub fn upgrade(&self) -> Option<ProgressBar> {
        self.state.upgrade().map(|state| ProgressBar { state })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_pbar_zero() {
        let pb = ProgressBar::new(0);
        assert_eq!(pb.state.lock().unwrap().fraction(), 1.0);
    }

    #[allow(clippy::float_cmp)]
    #[test]
    fn test_pbar_maxu64() {
        let pb = ProgressBar::new(!0);
        assert_eq!(pb.state.lock().unwrap().fraction(), 0.0);
    }

    #[test]
    fn test_pbar_overflow() {
        let pb = ProgressBar::new(1);
        pb.set_draw_target(ProgressDrawTarget::hidden());
        pb.inc(2);
        pb.finish();
    }

    #[test]
    fn test_get_position() {
        let pb = ProgressBar::new(1);
        pb.set_draw_target(ProgressDrawTarget::hidden());
        pb.inc(2);
        let pos = pb.position();
        assert_eq!(pos, 2);
    }

    #[test]
    fn test_weak_pb() {
        let pb = ProgressBar::new(0);
        let weak = pb.downgrade();
        assert!(weak.upgrade().is_some());
        ::std::mem::drop(pb);
        assert!(weak.upgrade().is_none());
    }

    #[test]
    fn test_draw_delta_deadlock() {
        // see issue #187
        let mpb = MultiProgress::new();
        let pb = mpb.add(ProgressBar::new(1));
        pb.set_draw_delta(2);
        drop(pb);
        mpb.join().unwrap();
    }

    #[test]
    fn test_abandon_deadlock() {
        let mpb = MultiProgress::new();
        let pb = mpb.add(ProgressBar::new(1));
        pb.set_draw_delta(2);
        pb.abandon();
        drop(pb);
        mpb.join().unwrap();
    }

    #[test]
    fn late_pb_drop() {
        let pb = ProgressBar::new(10);
        let mpb = MultiProgress::new();
        // This clone call is required to trigger a now fixed bug.
        // See <https://github.com/mitsuhiko/indicatif/pull/141> for context
        #[allow(clippy::redundant_clone)]
        mpb.add(pb.clone());
    }

    #[test]
    fn it_can_wrap_a_reader() {
        let bytes = &b"I am an implementation of io::Read"[..];
        let pb = ProgressBar::new(bytes.len() as u64);
        let mut reader = pb.wrap_read(bytes);
        let mut writer = Vec::new();
        io::copy(&mut reader, &mut writer).unwrap();
        assert_eq!(writer, bytes);
    }

    #[test]
    fn it_can_wrap_a_writer() {
        let bytes = b"implementation of io::Read";
        let mut reader = &bytes[..];
        let pb = ProgressBar::new(bytes.len() as u64);
        let writer = Vec::new();
        let mut writer = pb.wrap_write(writer);
        io::copy(&mut reader, &mut writer).unwrap();
        assert_eq!(writer.it, bytes);
    }

    #[test]
    fn progress_bar_sync_send() {
        let _: Box<dyn Sync> = Box::new(ProgressBar::new(1));
        let _: Box<dyn Send> = Box::new(ProgressBar::new(1));
        let _: Box<dyn Sync> = Box::new(MultiProgress::new());
        let _: Box<dyn Send> = Box::new(MultiProgress::new());
    }

    #[test]
    fn multi_progress_modifications() {
        let mp = MultiProgress::new();
        let p0 = mp.add(ProgressBar::new(1));
        let p1 = mp.add(ProgressBar::new(1));
        let p2 = mp.add(ProgressBar::new(1));
        let p3 = mp.add(ProgressBar::new(1));
        mp.remove(&p2);
        mp.remove(&p1);
        let p4 = mp.insert(1, ProgressBar::new(1));

        let state = mp.state.read().unwrap();
        // the removed place for p1 is reused
        assert_eq!(state.objects.len(), 4);
        assert_eq!(state.objects.iter().filter(|o| o.is_some()).count(), 3);

        // free_set may contain 1 or 2
        match state.free_set.last() {
            Some(1) => {
                assert_eq!(state.ordering, vec![0, 2, 3]);
                assert_eq!(extract_index(&p4), 2);
            }
            Some(2) => {
                assert_eq!(state.ordering, vec![0, 1, 3]);
                assert_eq!(extract_index(&p4), 1);
            }
            _ => unreachable!(),
        }

        assert_eq!(extract_index(&p0), 0);
        assert_eq!(extract_index(&p1), 1);
        assert_eq!(extract_index(&p2), 2);
        assert_eq!(extract_index(&p3), 3);
    }

    #[test]
    fn multi_progress_multiple_remove() {
        let mp = MultiProgress::new();
        let p0 = mp.add(ProgressBar::new(1));
        let p1 = mp.add(ProgressBar::new(1));
        // double remove beyond the first one have no effect
        mp.remove(&p0);
        mp.remove(&p0);
        mp.remove(&p0);

        let state = mp.state.read().unwrap();
        // the removed place for p1 is reused
        assert_eq!(state.objects.len(), 2);
        assert_eq!(state.objects.iter().filter(|obj| obj.is_some()).count(), 1);
        assert_eq!(state.free_set.last(), Some(&0));

        assert_eq!(state.ordering, vec![1]);
        assert_eq!(extract_index(&p0), 0);
        assert_eq!(extract_index(&p1), 1);
    }

    fn extract_index(pb: &ProgressBar) -> usize {
        match pb.state.lock().unwrap().draw_target.kind {
            ProgressDrawTargetKind::Remote { idx, .. } => idx,
            _ => unreachable!(),
        }
    }

    #[test]
    fn multi_progress_hidden() {
        let mpb = MultiProgress::with_draw_target(ProgressDrawTarget::hidden());
        let pb = mpb.add(ProgressBar::new(123));
        pb.finish();
        mpb.join().unwrap();
    }
}
