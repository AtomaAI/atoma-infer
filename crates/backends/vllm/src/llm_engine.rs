#[cfg(feature = "cuda")]
use std::{
    sync::{Arc, RwLock},
    time::{Duration, Instant, SystemTime},
};

#[cfg(feature = "cuda")]
use futures::StreamExt;
#[cfg(feature = "cuda")]
use tokenizers::Tokenizer;
#[cfg(feature = "cuda")]
use tokio::sync::mpsc::UnboundedReceiver;
#[cfg(feature = "cuda")]
use tracing::{error, info, info_span, instrument, trace, Span};

#[cfg(feature = "cuda")]
use crate::{
    egress::{ClientState, ResponseSenders},
    error::EngineError,
    model_executor::ModelThreadDispatcher,
    output::{GenerateRequestOutput, GenerateStreamingOutput, StreamResponse},
    policy::FcfsPolicy,
    request::EngineRequest,
    scheduler::{Scheduler, SchedulerOutputs},
    sequence::{
        ExecuteModelRequest, Sequence, SequenceGroupMetadata, SequenceGroupOutput, SequenceOutput,
        SequenceStatus,
    },
    types::{ReadLock, WriteLock},
    validation::StoppingCriteriaParameters,
};

/// Time in milliseconds we wait until we schedule new received requests,
/// in case the `LlmEngine` was on halt.
#[cfg(feature = "cuda")]
const SCHEDULE_WAIT_PERIOD: u64 = 100;

/// `LlmEngine` - An asynchronous worker responsible for scheduling new requests
/// and communicating with the `ModelExecutor` service to send new requests
/// for continuously batched AI inference.
#[cfg(feature = "cuda")]
pub struct LlmEngine {
    /// Dispatcher for communicating with the model executor's running thread,
    /// responsible for running prefill and decoding inference to produce AI-generated outputs.
    model_thread_dispatcher: ModelThreadDispatcher,
    /// Channel for receiving new requests from the running main `LlmService` instance.
    request_receiver: UnboundedReceiver<EngineRequest>,
    /// Client channels of the requests currently in flight, streaming and non-streaming alike.
    response_senders: ResponseSenders,
    /// Metadata of currently scheduled `SequenceGroup`s.
    sequence_groups_metadata: Vec<Arc<SequenceGroupMetadata>>,
    /// Current outputs from the scheduler.
    scheduler_outputs: SchedulerOutputs,
    /// Instance of the `Scheduler` with a First-Come-First-Serve policy.
    scheduler: Scheduler<FcfsPolicy>,
    /// Tokenizer for decoding sequences.
    tokenizer: Tokenizer,
    /// Tracing span for logging and monitoring.
    span: Span,
}

#[cfg(feature = "cuda")]
impl LlmEngine {
    /// Constructor
    pub fn new(
        model_thread_dispatcher: ModelThreadDispatcher,
        request_receiver: UnboundedReceiver<EngineRequest>,
        scheduler: Scheduler<FcfsPolicy>,
        tokenizer: Tokenizer,
    ) -> Self {
        Self {
            model_thread_dispatcher,
            sequence_groups_metadata: vec![],
            scheduler_outputs: SchedulerOutputs::create_empty(),
            scheduler,
            tokenizer,
            request_receiver,
            response_senders: ResponseSenders::default(),
            span: info_span!("llm-engine"),
        }
    }

    /// Main loop of the `LlmEngine`.
    ///
    /// This loop performs the following tasks:
    /// 1. Listens for incoming `SequenceGroup` requests and adds them to the `Scheduler`.
    /// 2. Waits for new outputs from the `ModelExecutor` service, processes these outputs, updates
    ///    the states of the associated `SequenceGroup`s, and re-schedules new requests.
    /// 3. Sends finished `SequenceGroup` outputs to the Atoma client service.
    ///
    /// The loop uses `tokio::select!` to concurrently handle incoming requests and model outputs.
    /// If there are no ongoing scheduled sequence groups, it waits for a short period before
    /// scheduling all received requests.
    #[instrument(skip(self))]
    pub async fn run(mut self) -> Result<(), EngineError> {
        let span = self.span.clone();
        let _enter = span.enter();

        loop {
            tokio::select! {
                Some(engine_request) = self.request_receiver.recv() => {
                    match engine_request {
                        EngineRequest::GenerateRequest(sequence_group, response_sender) => {
                            info!("Received new sequence group, with id = {}", sequence_group.request_id);
                            let sequence_group_request_id = sequence_group.request_id.clone();
                            // 1. Add the received `SequenceGroup` to the `Scheduler` instance.
                            self.scheduler.add_sequence_group(sequence_group);
                            // 2. Register the client waiting for this request's output.
                            self.response_senders.register_completion(sequence_group_request_id, response_sender);
                        },
                        EngineRequest::GenerateStreamingRequest(sequence_group, response_sender) => {
                            info!("Received new sequence group, with id = {}", sequence_group.request_id);
                            let sequence_group_request_id = sequence_group.request_id.clone();
                            // 1. Add the received `SequenceGroup` to the `Scheduler` instance.
                            self.scheduler.add_sequence_group(sequence_group);
                            // 2. Register the client streaming this request's output.
                            self.response_senders.register_stream(sequence_group_request_id, response_sender);
                        },
                    }
                    // 3. If the current `LlmInstance` doesn't have any on-going
                    //    scheduled sequence groups, we wait some time and then
                    //    schedule all the received requests so far.
                    //    This includes the request added in 1.
                    if self.sequence_groups_metadata.is_empty() && self.scheduler_outputs.is_empty() {
                        tokio::time::sleep(Duration::from_millis(SCHEDULE_WAIT_PERIOD)).await;
                        self.step()?;
                    }
                },
                Some(outputs) = self.model_thread_dispatcher.responses.next() => {
                    self.handle_outputs(outputs.map_err(EngineError::RecvError)).await?;
                }
                else => {
                    continue;
                }
            }
        }
    }

    /// Handles newly AI generated `SequenceGroupOutput`'s.
    ///
    /// This method processes the outputs generated by the AI model, schedules new requests,
    /// and sends the finished outputs to the Atoma client service. It performs the following steps:
    /// 1. Processes the newly AI generated outputs.
    /// 2. Schedules new requests.
    /// 3. Sends the finished outputs to the Atoma client service.
    ///
    /// If an error occurs while processing the outputs, it logs the error and continues to
    /// schedule new requests to maintain the system's liveness.
    ///
    /// # Arguments
    ///
    /// * `outputs` - A `Result` containing a vector of `SequenceGroupOutput` on success, or an
    ///   `EngineError` on failure.
    ///
    /// # Returns
    ///
    /// * `Result<(), EngineError>` - Returns `Ok(())` if the outputs are handled successfully, or
    ///   an `EngineError` if an error occurs.
    #[instrument(skip_all)]
    async fn handle_outputs(
        &mut self,
        outputs: Result<Vec<SequenceGroupOutput>, EngineError>,
    ) -> Result<(), EngineError> {
        let span = self.span.clone();
        let _enter = span.enter();

        match outputs {
            Ok(outputs) => {
                // 1. Processes the newly AI generated outputs
                let request_outputs = self.process_generated_outputs(outputs)?;

                // 2. Schedules new requests
                self.step()?;

                // 3. After scheduling new requests to the `ModelExecutor` we can send the finished
                //    outputs back to the OpenAI API service.
                // NOTE: This is after scheduling new sequences above,
                //    we do so to optimize GPU utilization. This is
                //    supposed to be safe
                //
                // A client that hung up before its output was ready costs that request its
                // channels and nothing more: the other outputs in this batch still go out.
                for request_output in request_outputs {
                    self.response_senders.complete(request_output);
                }
            }
            Err(e) => {
                error!("Invalid generated outputs with error: {e}");
                // NOTE: In order to maintain the system live, we need to keep calling
                // the `self.step()` method, even in possible scenarios of failure.
                self.step()?;
            }
        }
        Ok(())
    }

    /// Main scheduling method of `LlmEngine`.
    ///
    /// This method performs the following tasks:
    /// 1. Schedules new requests using the associated `Scheduler`.
    /// 2. Updates internal state with the new scheduling information.
    /// 3. If there are scheduled requests, creates and sends a new `ExecuteModelRequest` to the
    ///    `ModelExecutor`'s thread.
    ///
    /// # Returns
    /// - `Ok(())` if the scheduling and request sending are successful.
    #[instrument(skip_all)]
    pub fn step(&mut self) -> Result<(), EngineError> {
        let span = self.span.clone();
        let _enter = span.enter();

        trace!("`LlmEngine` new step..");
        // 1. Schedule new requests
        let (sequence_groups_metadata, scheduler_outputs) = self.scheduler.schedule()?;

        // 2. Update `self.scheduler_groups_metadata` and `scheduler_outputs`
        self.sequence_groups_metadata = sequence_groups_metadata.clone();
        self.scheduler_outputs = scheduler_outputs.clone();

        // 3. If the scheduled data is empty, it means that no new requests were received.
        if scheduler_outputs.is_empty() {
            return Ok(());
        }

        let execute_model_request = ExecuteModelRequest::new(
            sequence_groups_metadata,
            scheduler_outputs.blocks_to_swap_in,
            scheduler_outputs.blocks_to_swap_out,
            scheduler_outputs.blocks_to_copy,
            scheduler_outputs.running_queue_size,
        );

        // 4. Sends a new `ExecuteModelRequest` to the underlying `ModelExecutor`'s thread
        self.model_thread_dispatcher.send(execute_model_request);

        Ok(())
    }

    /// Processes newly generated AI outputs for sequence groups
    ///
    /// This function performs the following tasks:
    /// 1. Updates the state of each sequence in the scheduled sequence groups
    /// 2. Records metrics for sequence group processing times
    /// 3. Frees finished sequence groups
    /// 4. Aborts the requests whose client disconnected mid-generation
    /// 5. Collects and returns outputs for finished sequence groups
    ///
    /// # Arguments
    ///
    /// * `outputs` - A vector of `SequenceGroupOutput` containing the generated outputs
    ///
    /// # Returns
    ///
    /// * `Result<Vec<GenerateRequestOutput>, EngineError>` - A vector of `GenerateRequestOutput`
    ///   for finished sequence groups, or an error if processing fails
    #[instrument(skip_all)]
    fn process_generated_outputs(
        &mut self,
        outputs: Vec<SequenceGroupOutput>,
    ) -> Result<Vec<GenerateRequestOutput>, EngineError> {
        let now = Instant::now();
        let mut disconnected_requests = Vec::new();

        for (output, (sequence_group_metadata, scheduled_sequence_group)) in outputs.iter().zip(
            self.sequence_groups_metadata
                .iter()
                .zip(self.scheduler_outputs.scheduled_sequence_groups.iter()),
        ) {
            // 1. Update the number of computed tokens for scheduled `SequenceGroup`
            scheduled_sequence_group
                .scheduled_group
                .update_num_computed_tokens(scheduled_sequence_group.token_chunk_size)?;

            let stopping_criteria_params =
                scheduled_sequence_group.scheduled_group.stopping_params();

            // 2. Iterate over each `Sequence`s of `ScheduledSequenceGroup` and update its current
            //    state
            // after the new LLM inference iteration has been performed
            for (sequence_id, sequence) in scheduled_sequence_group.scheduled_group.sequences.iter()
            {
                let sequence_output = if let Some(output) = output.outputs.get(sequence_id) {
                    output
                } else {
                    error!(
                        "Missing generated sequence output token for sequence with id = {}",
                        sequence_id
                    );
                    return Err(EngineError::MissingSequenceOutputToken(*sequence_id));
                };

                // 3. Updates the state of the current `Sequence`
                let client_state = self.update_sequence(
                    sequence,
                    sequence_output,
                    sequence_group_metadata,
                    &stopping_criteria_params,
                )?;
                if client_state == ClientState::Disconnected {
                    disconnected_requests.push(sequence_group_metadata.request_id.clone());
                }
            }

            // 4. Add a few metrics
            let metrics_guard = scheduled_sequence_group
                .scheduled_group
                .metrics
                .read()
                .unwrap();

            let arrival_time_histogram = metrics::histogram!("sequence-group-arrival-time");
            arrival_time_histogram.record(metrics_guard.arrival_time.elapsed().as_secs_f32());

            let last_token_time_histogram = metrics::histogram!("sequence-group-last-token-time");
            last_token_time_histogram.record(metrics_guard.last_token_time.elapsed().as_secs_f32());
        }

        // 5. Removes all finished sequence groups from the `Scheduler`, returning their blocks
        self.scheduler.remove_finished_sequences()?;

        // 6. Retire the requests whose client hung up: nobody is reading their output, so the
        //    blocks they still hold have to go back to the pool.
        for request_id in disconnected_requests {
            info!("Aborting request {request_id}, its client disconnected");
            self.response_senders.remove(&request_id);
            self.scheduler.abort_sequence_group(request_id)?;
        }

        // 7. Keep track of all the finished `SequenceGroup`s
        let mut request_outputs = Vec::new();
        for scheduled_sequence_group in self.scheduler_outputs.scheduled_sequence_groups.iter() {
            scheduled_sequence_group
                .scheduled_group
                .maybe_set_first_scheduled_time(now);

            if scheduled_sequence_group.scheduled_group.is_finished() {
                request_outputs.push(GenerateRequestOutput::from_sequence_group(
                    &scheduled_sequence_group.scheduled_group,
                ));
            }
        }
        for sequence_group in self.scheduler_outputs.ignored_seq_groups.iter() {
            sequence_group.maybe_set_first_scheduled_time(now);
        }

        Ok(request_outputs)
    }

    /// Updates the state of a `Sequence` after an LLM inference iteration
    ///
    /// This method handles both the decoding phase (when generating new tokens) and the prefill
    /// phase.
    ///
    /// # Arguments
    /// * `sequence` - The `Sequence` to update
    /// * `sequence_output` - The output from the LLM for this sequence
    /// * `sequence_group_metadata` - Metadata for the sequence group
    /// * `stopping_criteria_params` - Parameters for stopping criteria
    ///
    /// # Returns
    /// * `Result<ClientState, EngineError>` - whether the request's client is still listening, or
    ///   an error if something goes wrong. A client that hung up is reported rather than raised:
    ///   the caller retires that one request and keeps serving the rest of the batch.
    ///
    /// # Behavior
    /// - In decoding phase (do_sample == true):
    ///   1. Updates sequence with new token, logprobs, and cumulative probability
    ///   2. Decodes and appends new token to output text
    ///   3. Checks for stopping conditions (stop token, EOS, length limits)
    ///   4. Updates sequence status if stopping condition is met
    /// - In prefill phase (do_sample == false):
    ///   1. Only updates the sequence's output log probabilities
    #[instrument(skip_all)]
    fn update_sequence(
        &self,
        sequence: &Arc<RwLock<Sequence>>,
        sequence_output: &SequenceOutput,
        sequence_group_metadata: &SequenceGroupMetadata,
        stopping_criteria_params: &StoppingCriteriaParameters,
    ) -> Result<ClientState, EngineError> {
        let sequence_id = { sequence.read_lock()?.sequence_id() };
        let request_id = &sequence_group_metadata.request_id;
        let mut client_state = ClientState::Connected;
        // 1. Get the AI generated next output token id.
        let generated_token_id = sequence_output.output_token;
        let is_stop_token = sequence_output.is_stop_token;

        if sequence_group_metadata.do_sample {
            let mut sequence_guard_lock = sequence.write_lock()?;
            // NOTE: this means we are in decoding phase.
            // That is, we are generating new output tokens
            // and these should be added to the `Sequence`'s
            // state.

            // 2. Update the `Sequence`'s output log-probabilities.
            //
            // 3. Update the `Sequence`'s `SequenceData` cumulative probabilities, if we are in
            //    decoding phase.
            //
            // 4. Update the `Sequence`'s `SequenceData` output tokens, if we are in decoding phase.
            sequence_guard_lock
                .add_token_id(generated_token_id, sequence_output.logprob.clone())?;

            // 5. Decode the generated output token id.
            let token_ids = sequence_guard_lock.sequence_data.get_token_ids();
            let generated_text = self
                .tokenizer
                .decode(&token_ids, true)
                .map_err(|e| EngineError::TokenizerError(e.to_string()))?;

            // 7. If the request is a streaming request, we need to send the generated token to the
            //    client as soon as possible.
            let created = SystemTime::now()
                .duration_since(SystemTime::UNIX_EPOCH)
                .expect("Failed to get system duration")
                .as_millis() as u64;
            let streaming_output = GenerateStreamingOutput {
                request_id: request_id.clone(),
                created,
                finish_reason: None,
                logprobs: sequence_guard_lock.output_logprobs.clone(),
                num_prompt_tokens: sequence_guard_lock.prompt_token_ids().len(),
                num_completion_tokens: sequence_guard_lock.tokens.len(),
                output_text: generated_text.clone(),
            };
            client_state = self
                .response_senders
                .send_chunk(request_id, StreamResponse::Chunk(streaming_output));

            // 8. Update the `output_text` with the newly generated token, if in decoding phase.
            let generated_token = if sequence_guard_lock.tokens.last().is_some() {
                let start = sequence_guard_lock.output_text.chars().count();
                generated_text.chars().skip(start).collect::<String>()
            } else {
                let start = sequence_guard_lock.prompt.chars().count();
                generated_text.chars().skip(start).collect()
            };
            sequence_guard_lock.output_text.push_str(&generated_token);

            // 9. Check if the last generated token is a stop token. If so, update the `Sequence`'s
            //    `SequenceState` and the `stop_reason`, as well.
            if stopping_criteria_params
                .stop_sequences
                .contains(&generated_token)
            {
                info!("Current sequence with id = {sequence_id} has finished execution due to stopping token = {generated_token}");
                sequence_guard_lock.stop_reason = Some(generated_token_id);
                sequence_guard_lock.set_sequence_status(SequenceStatus::FinishedStopped)
            }

            // 10. Check if the current `Sequence` last generated token
            //    id equals to the `eos_token_id`, in which case the
            //    the `Sequence`'s status should become `FinishedStopped`.
            if is_stop_token && !stopping_criteria_params.ignore_eos_token {
                sequence_guard_lock.set_sequence_status(SequenceStatus::FinishedStopped)
            }

            // 11. Check if the `Sequence`'s length exceeds that of `SchedulerConfig`'s. If so,
            //     update the `Sequence`'s `SequenceStatus` to `FinishedLengthCapped`.
            let sequence_len = sequence_guard_lock.length();
            if sequence_len > self.scheduler.scheduler_config.max_model_len() {
                sequence_guard_lock.set_sequence_status(SequenceStatus::FinishedLengthCapped)
            }

            // 12. Check if the `Sequence`'s output length exceeds that of Request's
            //     `max_new_tokens`.
            let sequence_output_len = sequence_guard_lock.get_output_len();
            if sequence_output_len >= stopping_criteria_params.max_new_tokens as usize {
                sequence_guard_lock.set_sequence_status(SequenceStatus::FinishedLengthCapped)
            }

            // 13. Tell a streaming client the sequence is done, whichever stopping criterion ended
            //     it, so the stream is closed exactly once.
            if sequence_guard_lock.is_finished()
                && self
                    .response_senders
                    .send_chunk(request_id, StreamResponse::Finished)
                    == ClientState::Disconnected
            {
                client_state = ClientState::Disconnected;
            }

            // 14. Update the `Sequence`'s tokens vec.
            sequence_guard_lock.tokens.push(generated_token)
        } else {
            // NOTE: in this case, we are not sampling newly
            // generated tokens. That is, we are in prefill
            // phase (possibly while chunking)
            // without generating the next token. For this reason,
            // we do not have to add tokens to the current
            // `Sequence`'s state.

            // 2. Update the `Sequence`'s output log-probabilities.
            sequence
                .write_lock()?
                .output_logprobs
                .push(sequence_output.logprob.clone());
        }

        Ok(client_state)
    }
}
