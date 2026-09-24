'use strict';

// Generated packet qualification only; no acquisition, firmware or model interface.
const assert = require('node:assert/strict');
const crypto = require('node:crypto');
const fs = require('node:fs');
const path = require('node:path');

const MIN = -(2 ** 23), MAX = 2 ** 23 - 1, U64 = 2n ** 64n - 1n;
const FLAGS = Object.freeze({ FINAL: 1, GAP: 2, RESET: 4, CONFIG: 8 });
const REASON = Object.freeze({ MISSING: 1, DISABLED: 2, SATURATION: 4,
  CALIBRATION_RANGE: 8, LEAD_OFF: 16, STATUS: 32 });
const sha = bytes => crypto.createHash('sha256').update(bytes).digest('hex');
function requireValue(ok, reason) { if (!ok) throw new Error(reason); }

function context(config, key, prefix) {
  requireValue(Buffer.isBuffer(key) && key.length === 16 && Buffer.isBuffer(prefix)
    && prefix.length === 8, 'KEY_OR_NONCE_PREFIX');
  requireValue(config && config.version === 1 && config.rate_hz === 250
    && config.channel_byte_order === 'big' && Array.isArray(config.channels)
    && config.channels.length === 4 && typeof config.capture_id === 'string'
    && config.capture_id.length > 0 && config.capture_id.length <= 128, 'CONFIGURATION');
  requireValue(Object.keys(config).every(k => ['version','rate_hz','channel_byte_order',
    'channels','capture_id','lead_p_mask','lead_n_mask'].includes(k)), 'UNKNOWN_CONFIGURATION_FIELD');
  requireValue(Number.isInteger(config.lead_p_mask) && config.lead_p_mask >= 0
    && config.lead_p_mask <= 15 && Number.isInteger(config.lead_n_mask)
    && config.lead_n_mask >= 0 && config.lead_n_mask <= 15, 'LEAD_CONFIGURATION');
  const channels = config.channels.map(c => {
    requireValue(c && typeof c.active === 'boolean' && typeof c.name === 'string'
      && c.name.length > 0 && c.name.length <= 128 && typeof c.calibration_id === 'string'
      && c.calibration_id.length > 0 && c.calibration_id.length <= 128
      && Number.isFinite(c.scale_uv_per_count) && c.scale_uv_per_count > 0
      && Number.isFinite(c.offset_count) && c.offset_count >= MIN && c.offset_count <= MAX
      && Number.isInteger(c.minimum_count) && Number.isInteger(c.maximum_count)
      && c.minimum_count >= MIN && c.maximum_count <= MAX && c.minimum_count < c.maximum_count
      && Number.isFinite(c.uncertainty_uv) && c.uncertainty_uv >= 0, 'CALIBRATION');
    requireValue(Object.keys(c).every(k => ['name','active','calibration_id','scale_uv_per_count',
      'offset_count','minimum_count','maximum_count','uncertainty_uv'].includes(k)), 'UNKNOWN_CALIBRATION_FIELD');
    for (const raw of [MIN, MAX]) requireValue(
      Number.isFinite(c.scale_uv_per_count * (raw - c.offset_count)), 'CALIBRATION_OVERFLOW');
    return Object.freeze({ name: c.name, active: c.active, calibration_id: c.calibration_id,
      scale_uv_per_count: c.scale_uv_per_count, offset_count: c.offset_count,
      minimum_count: c.minimum_count, maximum_count: c.maximum_count, uncertainty_uv: c.uncertainty_uv });
  });
  // This deterministic, closed metadata subset is the fixture's complete AAD configuration.
  const canonical = { version: 1, capture_id: config.capture_id, rate_hz: 250,
    channel_byte_order: 'big', lead_p_mask: config.lead_p_mask,
    lead_n_mask: config.lead_n_mask, channels: Object.freeze(channels) };
  const metadata = Buffer.from(JSON.stringify(canonical));
  return Object.freeze({ key: crypto.createSecretKey(key), prefix_hex: prefix.toString('hex'),
    config_sha256: sha(metadata), metadata_json: metadata.toString('utf8'),
    config: Object.freeze(canonical) });
}

function nonce(ctx, sequence) {
  requireValue(Number.isInteger(sequence) && sequence >= 0 && sequence <= 0xffffffff, 'SEQUENCE_RANGE');
  const out = Buffer.alloc(12);
  Buffer.from(ctx.prefix_hex, 'hex').copy(out, 0); out.writeUInt32BE(sequence, 8);
  return out;
}

function decodePacket(packet, ctx) {
  requireValue(Buffer.isBuffer(packet) && packet.length >= 55 && packet.length <= 340, 'PACKET_LENGTH');
  const header = packet.subarray(0, 24);
  const sequence = header.readUInt32LE(0), first = header.readBigUInt64LE(4);
  const tick = header.readBigUInt64LE(12), count = header.readUInt16LE(20);
  const version = header[22], flags = header[23];
  requireValue(count >= 1 && count <= 20 && packet.length === 24 + count * 15 + 16, 'PACKET_FRAMING');
  let plaintext;
  try {
    const dec = crypto.createDecipheriv('aes-128-ccm', ctx.key, nonce(ctx, sequence), { authTagLength: 16 });
    dec.setAuthTag(packet.subarray(-16));
    dec.setAAD(Buffer.concat([header, Buffer.from(ctx.config_sha256, 'hex')]), { plaintextLength: count * 15 });
    const part = dec.update(packet.subarray(24, -16));
    plaintext = Buffer.concat([part, dec.final()]);
  } catch { throw new Error('AUTHENTICATION_FAILED'); }
  requireValue(version === 1 && (flags & 0xf0) === 0, 'PACKET_VERSION_OR_FLAGS');
  requireValue(first + BigInt(count) - 1n <= U64, 'COUNTER_OVERFLOW');
  const raw = [], status = [], values = [], reasons = [];
  for (let i = 0; i < count; i++) {
    const s = plaintext.readUIntBE(i * 15, 3);
    const row = [], physical = [], why = [];
    const leadP = (s >>> 12) & 255, leadN = (s >>> 4) & 255;
    for (let ch = 0; ch < 4; ch++) {
      const c = ctx.config.channels[ch];
      const word = plaintext.readUIntBE(i * 15 + 3 + ch * 3, 3);
      const signed = word >= 0x800000 ? word - 0x1000000 : word;
      let reason = 0;
      if (!c.active) reason |= REASON.DISABLED;
      if (signed === MIN || signed === MAX) reason |= REASON.SATURATION;
      if (signed < c.minimum_count || signed > c.maximum_count) reason |= REASON.CALIBRATION_RANGE;
      if ((s >>> 20) !== 12) reason |= REASON.STATUS;
      if (((leadP & ctx.config.lead_p_mask) | (leadN & ctx.config.lead_n_mask)) & (1 << ch)) {
        reason |= REASON.LEAD_OFF;
      }
      row.push(signed); why.push(reason);
      physical.push(reason ? null : c.scale_uv_per_count * (signed - c.offset_count));
    }
    raw.push(row); status.push(s); values.push(physical); reasons.push(why);
  }
  return { sequence, first, tick, count, flags, raw, status, values, reasons, packet_sha256: sha(packet) };
}

function assemblePackets(packets, ctx, first, count) {
  requireValue(Array.isArray(packets) && packets.length <= 5000 && typeof first === 'bigint'
    && first >= 0n && first <= U64 && Number.isInteger(count) && count >= 1 && count <= 100000
    && first + BigInt(count) - 1n <= U64, 'CAPTURE_BOUNDS');
  const end = first + BigInt(count);
  const counts = Array.from({ length: count }, () => [0, 0, 0, 0]);
  const values = Array.from({ length: count }, () => [null, null, null, null]);
  const reasons = Array.from({ length: count }, () => [1, 1, 1, 1]);
  const statuses = Array(count).fill(null), errors = [], accepted = [];
  let previous = null, finalSeen = false;
  for (let index = 0; index < packets.length; index++) {
    const bytes = packets[index];
    let block;
    try { block = decodePacket(bytes, ctx); }
    catch (error) {
      errors.push({ packet_index: index, reason: error.message,
        packet_sha256: Buffer.isBuffer(bytes) ? sha(bytes) : null });
      continue;
    }
    requireValue(!finalSeen, 'DATA_AFTER_FINAL');
    requireValue(!(block.flags & (FLAGS.RESET | FLAGS.CONFIG)), 'NEW_SEGMENT_REQUIRED');
    requireValue(block.first >= first && block.first + BigInt(block.count) <= end, 'OUTSIDE_CAPTURE');
    const previousEnd = previous ? previous.first + BigInt(previous.count) : first;
    const sequenceGap = previous ? block.sequence - previous.sequence - 1 : block.sequence;
    const sampleGap = block.first - previousEnd;
    requireValue(sequenceGap >= 0 && sampleGap >= 0n
      && (!previous || block.tick > previous.tick), 'COUNTER_SEQUENCE_OR_TICK_ORDER');
    requireValue(sampleGap >= BigInt(sequenceGap), 'IMPOSSIBLE_MISSING_PACKET_EXTENT');
    requireValue(sampleGap > 0n || !(block.flags & FLAGS.GAP), 'FALSE_GAP_FLAG');
    requireValue(sampleGap === 0n || sequenceGap > 0 || (block.flags & FLAGS.GAP), 'UNDECLARED_SAMPLE_GAP');
    const start = Number(block.first - first);
    for (let i = 0; i < block.count; i++) {
      counts[start+i] = block.raw[i]; values[start+i] = block.values[i];
      reasons[start+i] = block.reasons[i]; statuses[start+i] = block.status[i];
    }
    if (block.flags & FLAGS.FINAL) {
      requireValue(block.first + BigInt(block.count) === end, 'FINAL_BOUNDARY_MISMATCH');
      finalSeen = true;
    }
    accepted.push({ packet_index: index, sequence: block.sequence, first_sample: block.first.toString(),
      first_tick: block.tick.toString(), count: block.count, flags: block.flags,
      packet_sha256: block.packet_sha256 });
    previous = block;
  }
  const valid = reasons.map(row => row.map(reason => reason === 0));
  const missing = reasons.filter(row => row[0] === REASON.MISSING).length;
  return { status: finalSeen && !errors.length && !missing ? 'AUTHENTICATED_TIMELINE_COMPLETE' : 'INCOMPLETE_CAPTURE',
    first_sample: first.toString(), sample_count: count, config_sha256: ctx.config_sha256,
    final_seen: finalSeen, missing_sample_slots: missing, counts, values_uv: values,
    valid, reasons, adc_status: statuses, accepted_packets: accepted, packet_errors: errors };
}

function sealFixture(ctx, { sequence = 0, first = 0n, tick = 0n, flags = 0, version = 1,
  samples, statuses = Array(samples.length).fill(0xc00000) }) {
  // Fixed test keys are public fixtures. This is not a persistent nonce-safe recorder.
  const header = Buffer.alloc(24);
  header.writeUInt32LE(sequence); header.writeBigUInt64LE(first, 4); header.writeBigUInt64LE(tick, 12);
  header.writeUInt16LE(samples.length, 20); header[22] = version; header[23] = flags;
  const plain = Buffer.alloc(samples.length * 15);
  samples.forEach((row, i) => {
    plain.writeUIntBE(statuses[i], i*15, 3);
    row.forEach((value, ch) => plain.writeUIntBE(value < 0 ? value + 0x1000000 : value, i*15+3+ch*3, 3));
  });
  const cipher = crypto.createCipheriv('aes-128-ccm', ctx.key, nonce(ctx, sequence), { authTagLength: 16 });
  cipher.setAAD(Buffer.concat([header, Buffer.from(ctx.config_sha256, 'hex')]), { plaintextLength: plain.length });
  const ciphertext = Buffer.concat([cipher.update(plain), cipher.final()]);
  return Buffer.concat([header, ciphertext, cipher.getAuthTag()]);
}

function fixtureConfig() {
  return { version: 1, capture_id: 'generated-packet-study', rate_hz: 250, channel_byte_order: 'big',
    lead_p_mask: 15, lead_n_mask: 15, channels: Array.from({ length: 4 }, (_, ch) => ({
      name: `synthetic-${ch}`, active: ch < 2, calibration_id: 'synthetic-affine-calibration',
      scale_uv_per_count: .25 * (ch+1), offset_count: ch-.5,
      minimum_count: -8000000, maximum_count: 8000000, uncertainty_uv: .1 })) };
}

function qualify(knownAnswer) {
  requireValue(process.version === 'v24.16.0' && process.versions.openssl === '3.5.6', 'RUNTIME_REQUALIFICATION_REQUIRED');
  const checks = [], config = fixtureConfig(), key = Buffer.alloc(16, 0x42), prefix = Buffer.alloc(8, 0x17);
  const ctx = context(config, key, prefix);
  const samples = [[MIN, -1, 0, MAX], [-8000000, 8000000, -123456, 123456]];
  const packet = sealFixture(ctx, { samples, first: 2n ** 53n + 17n, tick: 2n ** 60n + 19n });
  const decoded = decodePacket(packet, ctx);
  assert.equal(packet.subarray(0,24).toString('hex'), '000000001100000000002000130000000000001002000100');
  assert.equal(nonce(ctx,0x01020304).toString('hex'), '171717171717171701020304');
  assert.deepEqual(decoded.raw, samples);
  assert.equal(decoded.first, 2n ** 53n + 17n); assert.equal(decoded.tick, 2n ** 60n + 19n);
  assert.equal(decoded.values[0][1], -.75);
  assert.equal(decoded.values[0][0], null); assert.equal(decoded.values[0][2], null);
  checks.push('signed_24bit_extrema_and_counter_precision', 'calibrated_microvolts_and_disabled_channels');
  checks.push('literal_header_little_endian_and_nonce_big_endian');
  for (let n = 1; n <= 20; n++) {
    const p = sealFixture(ctx, { samples: Array.from({ length: n }, () => [1,-2,3,-4]) });
    assert.equal(p.length, 40+15*n); assert.equal(decodePacket(p, ctx).count, n);
  }
  checks.push('all_twenty_packet_lengths');
  const tamperPacket = sealFixture(ctx, { samples: Array.from({ length:20 }, () => [1,-2,3,-4]) });
  const flipped = [];
  for (let byte = 0; byte < tamperPacket.length; byte++) {
    const changed = Buffer.from(tamperPacket); changed[byte] ^= 1;
    assert.throws(() => decodePacket(changed, ctx)); flipped.push(byte);
  }
  for (let len = 0; len < tamperPacket.length; len++) assert.throws(() => decodePacket(tamperPacket.subarray(0, len), ctx));
  assert.throws(() => decodePacket(Buffer.concat([tamperPacket, Buffer.from([0])]), ctx));
  assert.throws(() => decodePacket(packet, context(config, Buffer.alloc(16), prefix)), /AUTHENTICATION/);
  assert.throws(() => decodePacket(packet, context(config, key, Buffer.alloc(8))), /AUTHENTICATION/);
  const revised = structuredClone(config); revised.channels[0].scale_uv_per_count *= 2;
  assert.throws(() => decodePacket(packet, context(revised, key, prefix)), /AUTHENTICATION/);
  checks.push('every_packet_byte_tamper_rejected', 'every_truncation_and_extra_byte_rejected', 'key_nonce_and_calibration_aad_bound');
  const lead = decodePacket(sealFixture(ctx, { samples: [[1,2,3,4]], statuses: [0xc02010] }), ctx);
  assert.equal(lead.reasons[0][0], REASON.LEAD_OFF); assert.equal(lead.reasons[0][1], REASON.LEAD_OFF);
  const badStatus = decodePacket(sealFixture(ctx, { samples: [[1,2,3,4]], statuses: [0] }), ctx);
  assert.ok(badStatus.reasons[0].every(x => (x & REASON.STATUS) !== 0));
  const range = decodePacket(sealFixture(ctx, { samples: [[8000001,-8000001,0,0]] }), ctx);
  assert.equal(range.reasons[0][0], REASON.CALIBRATION_RANGE);
  checks.push('lead_status_and_calibration_range_reason_flags');
  const unusedStatus = 0xc00000 | (0xf0 << 12) | (0xf0 << 4);
  const unused = decodePacket(sealFixture(ctx, { samples:[[1,2,3,4]], statuses:[unusedStatus] }),ctx);
  assert.deepEqual(unused.reasons[0],[0,0,REASON.DISABLED,REASON.DISABLED]);
  assert.equal(unused.status[0],unusedStatus);
  const senseOff = context({...config,lead_p_mask:0,lead_n_mask:0},key,prefix);
  const off = decodePacket(sealFixture(senseOff,{samples:[[1,2,3,4]],statuses:[0xcfffff]}),senseOff);
  assert.deepEqual(off.reasons[0],[0,0,REASON.DISABLED,REASON.DISABLED]);
  checks.push('disabled_and_unavailable_lead_status_preserved_without_inference');
  const row = Array.from({ length: 20 }, () => [123,-45,0,0]);
  const p0 = sealFixture(ctx, { samples: row });
  const p1 = sealFixture(ctx, { samples: row, sequence: 1, first: 20n, tick: 80000n });
  const p2 = sealFixture(ctx, { samples: row, sequence: 2, first: 40n, tick: 160000n, flags: FLAGS.FINAL });
  const complete = assemblePackets([p0,p1,p2], ctx, 0n, 60);
  assert.equal(complete.status, 'AUTHENTICATED_TIMELINE_COMPLETE');
  assert.equal(complete.values_uv[0][0], 30.875);
  const lost = assemblePackets([p0,p2], ctx, 0n, 60);
  assert.equal(lost.sample_count, 60); assert.equal(lost.missing_sample_slots, 20);
  assert.deepEqual(lost.values_uv.slice(20,40), Array.from({ length: 20 }, () => [null,null,null,null]));
  assert.deepEqual(lost.values_uv.slice(40), complete.values_uv.slice(40));
  const corrupt = Buffer.from(p1); corrupt[30] ^= 4;
  const failed = assemblePackets([p0,corrupt,p2], ctx, 0n, 60);
  assert.equal(failed.packet_errors[0].reason, 'AUTHENTICATION_FAILED');
  assert.equal(failed.missing_sample_slots, 20); assert.equal(failed.sample_count, 60);
  assert.equal(assemblePackets([p0], ctx, 0n, 60).final_seen, false);
  checks.push('complete_timeline', 'missing_and_unauthenticated_packets_preserve_slots', 'missing_final_is_incomplete');
  assert.throws(() => assemblePackets([p0,p0,p2], ctx, 0n, 60), /ORDER/);
  assert.throws(() => assemblePackets([p1,p0,p2], ctx, 0n, 60), /ORDER/);
  assert.throws(() => assemblePackets([p0,p1,p2,p0], ctx, 0n, 60), /AFTER_FINAL/);
  for (const flags of [FLAGS.RESET, FLAGS.CONFIG]) {
    const changed = sealFixture(ctx, { samples: row, flags });
    assert.throws(() => assemblePackets([changed], ctx, 0n, 20), /NEW_SEGMENT/);
  }
  for (const change of [{ flags: 16 }, { version: 2 }]) {
    assert.throws(() => decodePacket(sealFixture(ctx, { samples: row, ...change }), ctx), /VERSION_OR_FLAGS/);
  }
  assert.throws(() => decodePacket(sealFixture(ctx, { samples: row, first: U64 }), ctx), /OVERFLOW/);
  assert.throws(() => nonce(ctx, 2 ** 32), /SEQUENCE_RANGE/);
  checks.push('duplicate_reorder_and_post_final_rejected', 'resets_and_configuration_changes_require_new_segment',
    'unknown_schema_and_counter_wrap_rejected');
  const gap = sealFixture(ctx, { samples: row, sequence: 1, first: 30n, tick: 120000n, flags: FLAGS.GAP|FLAGS.FINAL });
  assert.equal(assemblePackets([p0,gap], ctx, 0n, 50).missing_sample_slots, 10);
  const hiddenGap = sealFixture(ctx, { samples: row, sequence: 1, first: 30n, tick: 120000n, flags: FLAGS.FINAL });
  assert.throws(() => assemblePackets([p0,hiddenGap], ctx, 0n, 50), /UNDECLARED_SAMPLE_GAP/);
  const falseGap = sealFixture(ctx, { samples: row, flags: FLAGS.GAP });
  assert.throws(() => assemblePackets([falseGap], ctx, 0n, 20), /FALSE_GAP/);
  checks.push('declared_acquisition_gaps_and_flags');
  for (const change of [{scale_uv_per_count:0},{scale_uv_per_count:Infinity},{scale_uv_per_count:Number.MAX_VALUE},
    {offset_count:NaN},{minimum_count:1,maximum_count:1},{uncertainty_uv:-1},{active:1},{calibration_id:''}]) {
    const invalid = structuredClone(config); Object.assign(invalid.channels[0], change);
    assert.throws(() => context(invalid,key,prefix), /CALIBRATION/);
  }
  checks.push('invalid_and_overflowing_calibration_rejected');
  const copyConfig = structuredClone(config), copyKey = Buffer.from(key), copyPrefix = Buffer.from(prefix);
  const copied = context(copyConfig,copyKey,copyPrefix);
  copyConfig.channels[0].scale_uv_per_count *= 10; copyKey.fill(0); copyPrefix.fill(0);
  assert.deepEqual(decodePacket(packet,copied).values,decoded.values);
  assert.throws(() => { copied.config.channels[0] = {}; }, TypeError);
  assert.throws(() => { copied.config.channels[0].offset_count = 55; }, TypeError);
  assert.throws(() => context({...config,unknown_field:1},key,prefix), /UNKNOWN_CONFIGURATION/);
  checks.push('context_immutable_and_source_buffers_copied');
  const lastShort = sealFixture(ctx,{samples:[[1,2,3,4]],sequence:2,first:40n,tick:160000n,flags:FLAGS.FINAL});
  assert.equal(assemblePackets([p0,p1,lastShort],ctx,0n,41).status,'AUTHENTICATED_TIMELINE_COMPLETE');
  const noPackets = assemblePackets([],ctx,0n,41);
  assert.equal(noPackets.missing_sample_slots,41); assert.equal(noPackets.status,'INCOMPLETE_CAPTURE');
  checks.push('short_final_and_empty_capture_preserve_declared_extent');
  JSON.parse(JSON.stringify(failed));
  assert.ok(!JSON.stringify(failed).includes('NaN'));
  checks.push('strict_json_safe_missing_values_and_bigint_metadata');
  let knownAnswerResult = null;
  if (knownAnswer) {
    const k = Buffer.from(knownAnswer.key_hex, 'hex'), n = Buffer.from(knownAnswer.nonce_hex, 'hex');
    const aad = Buffer.from(knownAnswer.aad_hex, 'hex'), plain = Buffer.from(knownAnswer.plaintext_hex, 'hex');
    requireValue(k.length === 16 && n.length === 12 && knownAnswer.expected_tag_hex.length === 32, 'KNOWN_ANSWER_PROFILE');
    const cipher = crypto.createCipheriv('aes-128-ccm', k, n, { authTagLength: 16 });
    cipher.setAAD(aad, { plaintextLength: plain.length });
    const encrypted = Buffer.concat([cipher.update(plain),cipher.final()]);
    assert.equal(encrypted.toString('hex'), knownAnswer.expected_ciphertext_hex);
    assert.equal(cipher.getAuthTag().toString('hex'), knownAnswer.expected_tag_hex);
    const dec = crypto.createDecipheriv('aes-128-ccm',k,n,{authTagLength:16});
    dec.setAuthTag(Buffer.from(knownAnswer.expected_tag_hex,'hex')); dec.setAAD(aad,{plaintextLength:encrypted.length});
    assert.deepEqual(Buffer.concat([dec.update(encrypted),dec.final()]),plain);
    knownAnswerResult = { passed: true, source: knownAnswer.source };
    checks.push('primary_source_aes128_nonce96_tag128_known_answer');
  }
  return { status: knownAnswer ? 'SYNTHETIC_PACKET_CHECKS_PASSED' : 'SYNTHETIC_CHECKS_WITHOUT_KNOWN_ANSWER',
    checks, byte_tamper_cases: flipped.length, node: process.version, openssl: process.versions.openssl,
    fixture_packet_sha256: sha(packet), fixture_metadata_sha256: ctx.config_sha256,
    known_answer: knownAnswerResult, missing_packet_sample_count: lost.missing_sample_slots,
    frozen_grid_sample_count: lost.sample_count, reason_flags: REASON,
    firmware_or_device_qualified: false, persistent_nonce_management_qualified: false,
    timeline_complete_is_not_signal_quality: true,
    product_route_enabled: false, clinical_validation: false };
}

module.exports = { context, nonce, decodePacket, assemblePackets, qualify, fixtureConfig, sealFixture, FLAGS, REASON };
if (require.main === module) {
  const folder = path.resolve(__dirname, '../reports/hardware-packet-design-v1');
  const output = path.join(folder, 'result.json'), vector = path.join(folder, 'known-answer.json');
  requireValue(!fs.existsSync(output), 'IMMUTABLE_EVIDENCE_EXISTS');
  requireValue(fs.existsSync(vector), 'PRIMARY_KNOWN_ANSWER_REQUIRED');
  const result = qualify(JSON.parse(fs.readFileSync(vector, 'utf8')));
  result.created_at = new Date().toISOString(); result.source_sha256 = sha(fs.readFileSync(__filename));
  result.known_answer_sha256 = sha(fs.readFileSync(vector));
  fs.mkdirSync(folder, { recursive: true });
  fs.writeFileSync(output, JSON.stringify(result, null, 2), { flag: 'wx' });
  console.log(JSON.stringify({ status: result.status, checks: result.checks.length, result: output }));
}
