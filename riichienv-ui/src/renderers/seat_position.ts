/** Four physical table edges: bottom, right, opposite, left.
 * Player indices are fixed seats. Sanma occupies edges 0, 1, 2 and leaves edge 3
 * empty; the dealer and seat winds change without moving players around the table.
 */
export function relativeSeat(player: number, viewpoint: number): number {
    return (player - viewpoint + 4) % 4;
}
