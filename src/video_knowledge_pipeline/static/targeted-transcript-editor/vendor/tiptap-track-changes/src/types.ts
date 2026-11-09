/*
 * Public-source curation marker, 2026-11-10.
 * Existing local VKP adaptation is retained below; this marker does not modify,
 * replace or re-date that adaptation.
 * Upstream base: sungkhum/tiptap-track-changes 0.2.1,
 * commit 25ca36690db5b4977e6d5db5911f9d1eebf6485f.
 * This file is part of a modified local source, not a byte-identical official
 * release. Original MIT copyright and license text
 * (vendor/tiptap-track-changes/LICENSE) remain retained.
 * No new authorship is claimed. The date above is the public-source curation
 * date only; it is not an inferred historical modification date.
 */

export interface ChangeAuthor {
  id: string;
  name: string;
  color: string;
}

export type TrackChangesMode = 'edit' | 'suggest' | 'view';

export interface TrackChangesOptions {
  author: ChangeAuthor;
  mode?: TrackChangesMode;
  onStatusChange?: (changeId: string, status: 'accepted' | 'rejected') => void;
  additionalBlockTypes?: string[];
}

export interface TrackChangesStorage {
  mode: TrackChangesMode;
  author: ChangeAuthor;
  onStatusChange?: (changeId: string, status: 'accepted' | 'rejected') => void;
}

export interface TrackedChangeInfo {
  changeId: string;
  type: 'insertion' | 'deletion' | 'formatChange' | 'nodeChange';
  authorId: string;
  authorName: string;
  authorColor: string;
  timestamp: string;
  from: number;
  to: number;
  text: string;
  formatAdded?: string;
  formatRemoved?: string;
}

export interface ChangeMarkAttributes {
  changeId: string;
  authorId: string;
  authorName: string;
  authorColor: string;
  timestamp: string;
}

export interface NodeChangeTracking {
  changeId: string;
  authorId: string;
  authorName: string;
  authorColor: string;
  timestamp: string;
  originalType: string;
  originalAttrs?: Record<string, unknown>;
}
